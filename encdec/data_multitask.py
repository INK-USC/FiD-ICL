import os
import json
import torch
import numpy as np

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from utils import load_data

class MultiTaskDataForEncDec(object):
    def __init__(self, logger, config, tokenizer, datasets, data_split, is_training):
        self.logger = logger
        self.config = config
        self.tokenizer = tokenizer
        self.datasets = datasets # the list of tasks to be loaded
        self.data_split = data_split
        self.is_training = is_training
    
    def __len__(self):
        return len(self.data)

    def load_raw_data(self):
        self.data = load_data(
            base_dir=self.config.data_dir, 
            datasets=self.datasets, 
            split=self.data_split, 
            k=self.config.k if self.is_training else self.config.test_k,
            seed=self.config.seed
        )

        self.logger.info("Printing 3 examples ...")
        for i in range(3):
            self.logger.info(self.data[i])

        # prepare metadata for evaluation
        self.metadata = []
        start_idx = 0
        for dp in self.data:
            n_options = len(dp["options"])
            if dp["output"] not in dp["options"]:
                print(dp)
            self.metadata.append({
                "task": dp["task"],
                "indices": list(range(start_idx, start_idx + n_options)),
                "options": dp["options"],
                "answer": dp["options"].index(dp["output"]),
            })
            start_idx += n_options

    def get_cache_path(self):
        cache_file = os.path.join(
            self.config.tensorize_dir, 
            "multitask_{}_split={}_k={}_seed={}_lower={}_maxlen={}_mode={}_istraining={}.json".format(
                self.config.task, 
                self.data_split,
                self.config.k, 
                self.config.seed, 
                self.config.lowercase,
                self.config.max_input_length,
                self.config.eval_mode,
                self.is_training
            )
        )
        return cache_file

    def load_dataset(self, use_cache=True):
        cache_file = self.get_cache_path()

        if os.path.exists(cache_file) and use_cache:
            self.logger.info("Loading tensorized data from {}".format(cache_file))
            with open(cache_file, "r") as fin:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = json.load(fin)

            assert len(input_ids) == len(decoder_input_ids)
            if self.is_training:
                assert len(input_ids) == len(self.data)
        else:
            self.logger.info("Tokenizing {} examples".format(len(self.data)))

            if self.is_training or self.config.eval_mode == "generation":
                inputs = [dp["input"] for dp in self.data]
                outputs = [dp["output"] for dp in self.data]
            else:
                # copy many time for rank classification
                inputs, outputs = [], []
                for dp in self.data:
                    inputs += [dp["input"]] * len(dp["options"])
                    outputs += dp["options"]

            if self.config.lowercase:
                inputs = [item.lower() for item in inputs]
                outputs = [item.lower() for item in outputs]

            if self.config.use_prefix_for_query_input:
                inputs = [self.config.input_prefix + " " + item + self.config.output_prefix for item in inputs]
                # NOTE:            
                # multiple whitesapce will be the same as one whitespace, so adding whitespace here just in case
                # ref: https://github.com/huggingface/transformers/issues/6150

            self.logger.info("Tokenizing Input ...")
            tokenized_input = self.tokenizer.batch_encode_plus(inputs,
                                                         padding='max_length',
                                                         truncation=True,
                                                         max_length=self.config.max_input_length,
                                                         add_special_tokens=self.config.add_special_tokens)
            self.logger.info("Tokenizing Output ...")
            tokenized_output = self.tokenizer.batch_encode_plus(outputs,
                                                       padding='max_length',
                                                       truncation=True,
                                                       max_length=self.config.max_input_length,
                                                       add_special_tokens=True)
            # regarding `add_special_tokens`:
            # P3 data input_ids in huggingface datasets does not have the pad_token at the end
            # but the target_ids still have the pad_token
            # to keep things consistent, we remove pad_token for T0-Evaluation

            input_ids, attention_mask = tokenized_input["input_ids"], tokenized_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = tokenized_output["input_ids"], tokenized_output["attention_mask"]
            
            if self.config.fid_special_tokens:
                # 32000 is <extra_id_99>
                # I use this to `highlight` and separate the last instance from the in-context instances
                input_ids = [[32000] + item for item in input_ids]
                attention_mask = [[1] + item for item in attention_mask]

            if use_cache:
                self.logger.info("Saving tensorized data to {}".format(cache_file))
                with open(cache_file, "w") as fout:
                    json.dump([input_ids, attention_mask, decoder_input_ids, decoder_attention_mask], fout)

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        decoder_input_ids = torch.LongTensor(decoder_input_ids)
        decoder_attention_mask = torch.LongTensor(decoder_attention_mask)

        # according to https://huggingface.co/docs/transformers/model_doc/t5
        # you need to mark pad tokens to be -100 manually
        # if self.is_training and self.config.train_with_generation_loss:
        #     decoder_input_ids [decoder_input_ids == self.tokenizer.pad_token_id] = -100
        
        self.dataset = TensorDataset(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)

    def load_dataloader(self):
        if self.is_training:
            sampler = RandomSampler(self.dataset)
            self.dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.config.batch_size)
        else:
            sampler = SequentialSampler(self.dataset)
            self.dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.config.eval_batch_size)

    def evaluate(self, all_predictions):
        assert self.is_training==False, len(self.data)==len(all_predictions)

        performance_dict = {}
        for dataset in self.datasets:
            datapoints, predictions = [], []
            for dp, prediction in zip(self.data, all_predictions):
                if dp["task"] in dataset: # a temparary hack for boolq vs. boolq-all
                    datapoints.append(dp)
                    predictions.append(prediction)
            metric, perf = self.evaluate_a_dataset(dataset, predictions, datapoints)
            performance_dict[dataset] = (metric, perf)

        if len(self.datasets) > 1: # multitask
            return performance_dict
        else: # singletask
            return metric, perf

    def evaluate_a_dataset(self, dataset_name, predictions, datapoints):
        # TODO(qinyuany): extend to non-accuracy evaluation (by reading from some external config files)
        is_classification = False 
        
        accs = []
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        for prediction, datapoint in zip(predictions, datapoints):
            prediction = prediction.strip()
            groundtruth = datapoint["output"].strip()
            is_correct = prediction==groundtruth
            accs.append(is_correct)
            if is_classification:
                recalls[groundtruth].append(is_correct)
                precisions[prediction].append(is_correct)

        if not is_classification:
            return "acc", np.mean(accs)

        f1s = []
        for key in recalls:
            precision = np.mean(precisions[key]) if key in precisions else 1.0
            recall = np.mean(recalls[key])
            if precision+recall==0:
                f1s.append(0)
            else:
                f1s.append(2*precision*recall / (precision+recall))

        return "f1", np.mean(f1s)
    

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, tokens):
        return [self.decode(_tokens).strip() for _tokens in tokens]

