import os
import json
import torch
import random

from data_singletask import SingleTaskDataForEncDec
from promptsource.templates import DatasetTemplates
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from task_configs.fid_eval import N_OPTIONS

class T0FewshotDataForEncDec(SingleTaskDataForEncDec):
    def load_raw_data(self, dataset, prompt, seed):

        clean_dataset = dataset.replace("/", "_")
        if clean_dataset.endswith("_2016"):
            clean_dataset = clean_dataset[:-5]
        file_path = "../t0/data_fewshot/{}/16_shot/{}_seed.jsonl".format(clean_dataset, seed)

        # load raw few-shot data
        with open(file_path, "r") as fin:
            self.data = []
            for idx, line in enumerate(fin.readlines()):
                self.data.append(json.loads(line.strip("\n")))

        # for perturbation experiments
        assert self.config.perturbation_mode in [None, "random_label", "wrong_label"]
        if self.config.perturbation_mode == "random_label":
            all_options = list(range(N_OPTIONS[dataset]))
            for item in self.data:
                item["label"] = random.choice(all_options) # randomly select 1
        if self.config.perturbation_mode == "wrong_label":
            all_options = list(range(N_OPTIONS[dataset]))
            for item in self.data:
                wrong_options = all_options.copy()
                wrong_options.remove(item["label"])
                item["label"] = random.choice(wrong_options)

        clean_dataset = "anli" if dataset.startswith("anli") else dataset
        prompt = DatasetTemplates(clean_dataset)[prompt]
        examples = [prompt.apply(item) for item in self.data]

        self.inputs = [result[0] for result in examples]
        self.outputs = [result[1] for result in examples]

        if self.config.varying_shots:
            self.inputs = self.inputs[:self.config.varying_shots]
            self.outputs = self.outputs[:self.config.varying_shots]
        
    def load_dataset(self):

        if self.config.lowercase:
            self.inputs = [item.lower() for item in inputs]
            self.outputs = [item.lower() for item in outputs]

        self.logger.info("Tokenizing Input ...")
        tokenized_input = self.tokenizer.batch_encode_plus(self.inputs,
                                                        padding='max_length',
                                                        truncation=True,
                                                        max_length=self.config.max_input_length,
                                                        add_special_tokens=self.config.add_special_tokens)
        self.logger.info("Tokenizing Output ...")
        tokenized_output = self.tokenizer.batch_encode_plus(self.outputs,
                                                    padding='max_length',
                                                    truncation=True,
                                                    max_length=self.config.max_input_length,
                                                    add_special_tokens=True)

        input_ids, attention_mask = tokenized_input["input_ids"], tokenized_input["attention_mask"]
        decoder_input_ids, decoder_attention_mask = tokenized_output["input_ids"], tokenized_output["attention_mask"]

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        decoder_input_ids = torch.LongTensor(decoder_input_ids)
        decoder_attention_mask = torch.LongTensor(decoder_attention_mask)

        self.dataset = TensorDataset(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)

    def load_dataloader(self):
        assert self.is_training
        sampler = RandomSampler(self.dataset)
        self.dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.config.batch_size)
