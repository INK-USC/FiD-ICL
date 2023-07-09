import torch
import torch.nn.functional as F
import time

import numpy as np

from trainer import Trainer
from utils import trim_batch, label_smoothed_nll_loss
from tqdm import tqdm

class FiDTrainer(Trainer):

    def trim_batch(self, batch, pad_token_id):
        batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
        if len(batch) > 2:
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])
            if len(batch) > 4:
                batch[4], batch[5] = trim_batch(batch[4], pad_token_id, batch[5])
        return batch

    def print_tensor(self, tensor):
        # print the examples in a tensor
        for i, ids in enumerate(tensor.cpu().tolist()):
            _text = self.tokenizer.decode(ids)
            self.logger.info("Example {}".format(i))
            self.logger.info("Text: {}".format(_text))
                
    def print_batch(self, batch):
        # for debugging purposes

        # few-shot (support) examples
        self.logger.info("--- Support examples:")
        for i, ids in enumerate(batch[0].cpu().tolist()):
            _text = self.tokenizer.decode(ids)
            self.logger.info("Example {}".format(i))
            self.logger.info("Text: {}".format(_text))
        
        # query examples
        self.logger.info("--- Query examples:")
        for i, (input_ids, target_ids) in enumerate(zip(batch[2].cpu().tolist(), batch[4].cpu().tolist())):
            _input = self.tokenizer.decode(input_ids)
            _output = self.tokenizer.decode(target_ids)
            self.logger.info("Example {}".format(i))
            self.logger.info("Input: {}".format(_input))
            self.logger.info("Output: {}".format(_output))
        

    def run_model(self, model, batch, is_training=True):

        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]

        batch = self.trim_batch(batch, self.pad_token_id)

        # self.print_batch(batch)
        # breakpoint()

        support_input_ids, support_attention_mask = batch[0], batch[1]  # k * src_len
        query_input_ids, query_attention_mask = batch[2], batch[3] # m * src_len
        target_input_ids, target_attention_mask = batch[4], batch[5] # m * tar_len

        # for training, set this to be ignore_index
        target_input_ids [target_input_ids == self.pad_token_id] = -100

        k, m, m_ = support_input_ids.shape[0], query_input_ids.shape[0], target_input_ids.shape[0]
        assert m == m_

        support_hidden_states = model.encoder(input_ids=support_input_ids, attention_mask=support_attention_mask).last_hidden_state
        # support_hidden_states: k * src_len * hidden_dim
        support_hidden_states = support_hidden_states.flatten(0,1)
        # support_hidden_states: (k * src_len) * hidden_dim
        support_attention_mask = support_attention_mask.flatten(0,1)
        # support_attention_mask: (k * src_len)

        # remove padding positions
        support_hidden_states = support_hidden_states[support_attention_mask.bool(), :]
        # support_hidden_states: (new_length) * hidden_dim
        support_attention_mask = torch.ones(
            1, support_attention_mask.sum(), 
            dtype=torch.long, device=support_attention_mask.device
        )

        # repeat to the bsz of query
        support_hidden_states = support_hidden_states.unsqueeze(0).expand(m, -1, -1)
        support_attention_mask = support_attention_mask.expand(m, -1)
        # support_hidden_states: m * new_length * hidden_dim
        # support_attention_mask: m * new_length

        query_hidden_states = model.encoder(input_ids=query_input_ids, attention_mask=query_attention_mask).last_hidden_state
        # query_hidden_states: m * src_len * hidden_dim

        # concat w.r.t sequence length
        all_hidden_states = torch.cat([support_hidden_states, query_hidden_states], dim=1)
        all_attention_mask = torch.cat([support_attention_mask, query_attention_mask], dim=1)

        if is_training:
            target_input_ids [target_input_ids == self.pad_token_id] = -100

        model_output = model(
            attention_mask=all_attention_mask,
            encoder_outputs=[all_hidden_states],
            labels=target_input_ids,
            decoder_attention_mask=target_attention_mask,
            use_cache=False
        )

        loss = model_output.loss
        # print(loss)
        # breakpoint()
        return loss

    # ideally precompute_support_hidden_states + run_model_eval should be similar to run_model
    def precompute_support_hidden_states(self, model, batch):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]

        batch = self.trim_batch(batch, 0)

        support_input_ids = batch[0]
        support_attention_mask = batch[1]

        support_hidden_states = model.encoder(input_ids=support_input_ids, attention_mask=support_attention_mask).last_hidden_state
        # support_hidden_states: k * src_len * hidden_dim
        support_hidden_states = support_hidden_states.flatten(0,1)
        # support_hidden_states: (k * src_len) * hidden_dim
        support_attention_mask = support_attention_mask.flatten(0,1)
        # support_attention_mask: (k * src_len)

        # remove padding positions
        support_hidden_states = support_hidden_states[support_attention_mask.bool(), :]
        # support_hidden_states: (new_length) * hidden_dim
        support_attention_mask = torch.ones(
            1, support_attention_mask.sum(), 
            dtype=torch.long, device=support_attention_mask.device
        )

        return support_hidden_states, support_attention_mask

    def run_model_eval(self, model, batch, support_hidden_states, support_attention_mask):
        # assuming that support_hidden_states, support_attention_mask are pre-computed
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]

        batch = self.trim_batch(batch, 0)

        query_input_ids, query_attention_mask = batch[0], batch[1] # m * src_len
        target_input_ids, target_attention_mask = batch[2], batch[3] # m * tar_len

        m = query_input_ids.shape[0]
        
        # repeat to the bsz of query
        support_hidden_states = support_hidden_states.unsqueeze(0).expand(m, -1, -1)
        support_attention_mask = support_attention_mask.expand(m, -1)
        # support_hidden_states: m * new_length * hidden_dim
        # support_attention_mask: m * new_length

        query_hidden_states = model.encoder(input_ids=query_input_ids, attention_mask=query_attention_mask).last_hidden_state

        # concat w.r.t sequence length
        all_hidden_states = torch.cat([support_hidden_states, query_hidden_states], dim=1)
        all_attention_mask = torch.cat([support_attention_mask, query_attention_mask], dim=1)

        model_output = model(
            attention_mask=all_attention_mask,
            encoder_outputs=[all_hidden_states],
            labels=target_input_ids,
            decoder_attention_mask=target_attention_mask,
            use_cache=False
        )

        # model_output = model(
        #     input_ids=query_input_ids,
        #     attention_mask=query_attention_mask,
        #     labels=target_input_ids,
        #     decoder_attention_mask=target_attention_mask,
        # )

        # rank_classification
        lprobs = F.log_softmax(model_output.logits, dim=-1)
        loss, _ = label_smoothed_nll_loss(
            lprobs, target_input_ids, 
            epsilon=0.0, 
            ignore_index=model.config.pad_token_id,
            average=self.config.loss_avg_mode, # by default it's per-instance token-avg loss
        )

        return loss

    def run_model_eval2(self, model, batch, support_hidden_states, support_attention_mask):
        # assuming that support_hidden_states, support_attention_mask are pre-computed
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]

        batch = self.trim_batch(batch, 0)

        query_input_ids, query_attention_mask = batch[0], batch[1] # m * src_len
        target_input_ids, target_attention_mask = batch[2], batch[3] # m * tar_len

        m = query_input_ids.shape[0]
        
        # repeat to the bsz of query
        support_hidden_states = support_hidden_states.unsqueeze(0).expand(m, -1, -1)
        support_attention_mask = support_attention_mask.expand(m, -1)
        # support_hidden_states: m * new_length * hidden_dim
        # support_attention_mask: m * new_length

        query_hidden_states = model.encoder(input_ids=query_input_ids, attention_mask=query_attention_mask).last_hidden_state

        # concat w.r.t sequence length
        all_hidden_states = torch.cat([support_hidden_states, query_hidden_states], dim=1)
        all_attention_mask = torch.cat([support_attention_mask, query_attention_mask], dim=1)

        model_output = model(
            attention_mask=all_attention_mask,
            encoder_outputs=[all_hidden_states],
            labels=target_input_ids,
            decoder_attention_mask=target_attention_mask,
            use_cache=False
        )

        # rank_classification
        lprobs = F.log_softmax(model_output.logits, dim=-1)
        loss, _ = label_smoothed_nll_loss(
            lprobs, target_input_ids, 
            epsilon=0.0, 
            ignore_index=model.config.pad_token_id,
            average=self.config.loss_avg_mode, # by default it's per-instance token-avg loss
        )

        return loss

    def do_eval(self, model, data):
        model.eval()
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        losses = []

        batch = [data.support_input_ids, data.support_attention_mask]

        start_time = time.time()
        support_hidden_states, support_attention_mask = self.precompute_support_hidden_states(model, batch)
        end_time = time.time()
        # self.logger.info("precompute time {}".format(end_time - start_time))

        for batch in tqdm(data.dataloader, desc="Eval (Rank)"):
            with torch.no_grad():
                # batch = self.trim_batch(batch, pad_token_id=data.tokenizer.pad_token_id)
                loss = self.run_model_eval(model, batch, support_hidden_states, support_attention_mask)
                losses += loss.cpu().detach().numpy().tolist()
        losses = np.array(losses)

        predictions = []
        for idx, dp in enumerate(data.metadata):
            curr_instance_losses = [losses[indices] for indices in dp["indices"]]
            prediction_idx = sorted(enumerate(curr_instance_losses), key=lambda x: x[1])[0][0]
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())

        perf = data.evaluate(predictions)
        self.logger.info("Evaluation results: {}".format(perf))
        return perf