import torch
import torch.nn.functional as F
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup, AutoModelForSeq2SeqLM
from tqdm import tqdm

from utils import trim_batch, label_smoothed_nll_loss

class Trainer(object):
    def __init__(self, config, logger, local_rank=-1):
        self.config = config
        self.logger = logger
        self.pad_token_id = None
        self.local_rank = local_rank

    def load_model(self, path=None):
        if path is not None:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model, state_dict=torch.load(path))
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model)

        if self.config.do_train and (self.config.model == "bigscience/T0_3B" or self.config.model == "google/t5-xl-lm-adapt"):
            model.gradient_checkpointing_enable()
        if self.config.do_train and self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        return model

    def save(self, model, postfix):
        if self.local_rank <= 0:
            model_state_dict = {
                key: value.cpu() 
                for key, value in model.state_dict().items()
            }
            torch.save(model_state_dict, os.path.join(self.config.out_dir, "model-{}.pt".format(postfix)))
            self.logger.info("Saving model with postfix {}".format(postfix))

    def setup_optimizer(self, model):
        config = self.config
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if self.config.optimizer == "adamw":

            optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=config.warmup_steps,
                                            num_training_steps=config.num_training_steps)
        elif self.config.optimizer == "adafactor":
            optimizer = Adafactor(optimizer_grouped_parameters,
                                  lr=config.lr,
                                  relative_step=False,
                                  clip_threshold=config.max_grad_norm,
                                  warmup_init=False)
            scheduler = None
        return optimizer, scheduler

    def log(self, tb, log_dict, step):
        for key, value in log_dict.items():
            tb.add_scalar(key, value, step)

    def print_tensor(self, input_tensor, output_tensor):
        # print the examples in a tensor
        for i, (in_ids, out_ids) in enumerate(zip(input_tensor.cpu().tolist(), output_tensor.cpu().tolist())):
            input_text = self.tokenizer.decode(in_ids)
            output_text = self.tokenizer.decode(out_ids)
            self.logger.info("Example {}".format(i))
            self.logger.info("Input: {}".format(input_text))
            self.logger.info("Output: {}".format(output_text))

    def do_train(self, model, data, dev_data=None):
        self.tokenizer = data.tokenizer # will be used by some functions (e.g., `print_batch` in FiDTrainer)

        if self.config.use_tensorboard:
            tb_writer = SummaryWriter(log_dir=self.config.out_dir)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        model.train()

        optimizer, scheduler = self.setup_optimizer(model)

        global_step = 0
        global_batch = 0
        train_losses = []
        grad_norms = []
        best_perf = -1
        stop_training = False

        pbar = tqdm(total=self.config.num_training_steps)
        for epoch in range(1000000):
            pbar.set_description("Epoch {}".format(epoch))
            for batch in data.dataloader:
                global_batch += 1

                # truncate the redundant padding tokens
                # batch = self.trim_batch(batch, pad_token_id=data.tokenizer.pad_token_id)
                loss = self.run_model(model, batch)

                if torch.isnan(loss).data:
                    self.logger.info("Stop training because loss=%s" % (loss.data))
                    stop_training = True
                    break

                train_losses.append(loss.detach().cpu())
                loss.backward()

                if global_batch % self.config.gradient_accumulation_steps == 0:
                    global_step += 1
                    pbar.update(1)

                    if self.config.max_grad_norm is not None and self.config.optimizer != "adafactor":
                        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                        grad_norms.append(gn.detach().cpu())

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    model.zero_grad()

                    if self.config.do_valid and global_step % self.config.valid_period == 0:
                        assert dev_data is not None
                        metric, perf = self.do_eval(model, dev_data)

                        self.logger.info("Validation at step {}: {}={}".format(global_step, metric, perf))
                        
                        if self.config.use_tensorboard:
                            log_dict = {"dev_performance": perf}
                            self.log(tb_writer, log_dict, global_step)
                            
                        if perf > best_perf:
                            self.logger.info("Saving the best model so far ({}: {} --> {})".format(metric, best_perf, perf))
                            self.save(model, "best")
                            best_perf = perf

                        model.train() # do_eval switches the model to eval mode; here we switch back
                
                    if self.config.use_tensorboard and global_step % self.config.log_period == 0:
                        log_dict = {"loss": np.mean(train_losses)}
                        train_losses = []
                        if len(grad_norms) > 0:
                            log_dict["grad_norm"] = np.mean(grad_norms)
                            grad_norms = []
                        self.log(tb_writer, log_dict, global_step)

                    if self.config.save and global_step % self.config.save_period == 0:
                        self.save(model, str(global_step))

                    if global_step==self.config.num_training_steps or stop_training:
                        break

            if global_step==self.config.num_training_steps or stop_training:
                break

        pbar.close()
        self.logger.info("Finish training")
        
        if self.config.save:
            self.save(model, "last")

        return best_perf

    def do_eval(self, model, data):
        model.eval()
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        assert self.config.eval_mode in ["rank_classification", "generation"]

        if self.config.eval_mode == "rank_classification":
            predictions = self.do_eval_rank_classification(model, data)
            # from collections import Counter
            # counter = Counter(predictions)
            # print(counter)
        elif self.config.eval_mode == "generation":
            predictions = self.do_eval_generation(model, data)
            
        perf = data.evaluate(predictions)
        self.logger.info("Evaluation results: {}".format(perf))
        return perf

    def do_eval_rank_classification(self, model, data):
        losses = []
        for batch in tqdm(data.dataloader, desc="Eval (Rank)"):
            with torch.no_grad():
                # self.logger.info(batch[0])
                # self.print_tensor(batch[0], batch[2])
                # breakpoint()
                loss = self.run_model(model, batch, is_training=False)
                losses += loss.cpu().detach().numpy().tolist()
        losses = np.array(losses)

        predictions = []
        for idx, dp in enumerate(data.metadata):
            curr_instance_losses = [losses[indices] for indices in dp["indices"]]
            prediction_idx = sorted(enumerate(curr_instance_losses), key=lambda x: x[1])[0][0]
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())

        return predictions

    def do_eval_generation(self, model, data):
        pad_token_id = data.tokenizer.pad_token_id
        predictions = []
        for batch in tqdm(data.dataloader, desc="Eval (Generation)"):
            with torch.no_grad():
                if torch.cuda.is_available():
                    batch = [b.to(torch.device("cuda")) for b in batch]
                batch = self.trim_batch(batch, pad_token_id=data.tokenizer.pad_token_id)
                outputs = model.generate(input_ids=batch[0],
                                        attention_mask=batch[1],
                                        num_beams=4,
                                        max_length=64,
                                        early_stopping=True,
                                        use_cache=True)
                predictions += data.decode_batch(outputs)

        return predictions

    def trim_batch(self, batch, pad_token_id):
        batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
        if len(batch) == 4:
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])
        return batch        

    def run_model(self, model, batch, is_training=True):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]

        batch = self.trim_batch(batch, self.pad_token_id)
        input_ids, attention_mask = batch[0], batch[1]
        decoder_input_ids, decoder_attention_mask = batch[2], batch[3]


        if is_training and self.config.train_with_generation_loss:
            decoder_input_ids [decoder_input_ids == self.pad_token_id] = -100

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False
        )

        if is_training and self.config.train_with_generation_loss:
            return output.loss

        # rank_classification
        lprobs = F.log_softmax(output.logits, dim=-1)
        loss, _ = label_smoothed_nll_loss(
            lprobs, decoder_input_ids, 
            epsilon=0.0, 
            # epsilon=0.1 if is_training else 0.0, 
            ignore_index=model.config.pad_token_id,
            average=self.config.loss_avg_mode, # by default it's per-instance token-avg loss
        )

        if is_training:
            return loss.mean()
        else:
            return loss

