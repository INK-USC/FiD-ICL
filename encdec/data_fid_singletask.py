import os
import json
import torch
import random

from data_t0singletask import T0SingleTaskDataForEncDec, load_data
from data_fid import FiDPretrainDataForEncDec
from promptsource.templates import DatasetTemplates
from task_configs.fid_eval import N_OPTIONS

class FiDSingleTaskDataForEncDec(FiDPretrainDataForEncDec, T0SingleTaskDataForEncDec):
    def get_cache_path(self):
        cache_file = os.path.join(
            self.config.tensorize_dir, 
            "fid_singletask_{}_split={}_k={}_seed={}_lower={}_maxlen={}_mode={}_istraining={}.json".format(
                self.dataset, 
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

    def load_raw_data(self):
        self.data = load_data(
            base_dir=self.config.data_dir, 
            datasets=self.datasets, 
            split=self.data_split, 
        )

        self.logger.info("Printing 3 examples ...")
        for i in range(3):
            self.logger.info(self.data[i])

        # prepare metadata for evaluation
        self.metadata = []
        start_idx = 0
        for dp in self.data:
            n_options = len(dp["options"])
            self.metadata.append({
                "task": dp["task"],
                "indices": list(range(start_idx, start_idx + n_options)),
                "options": dp["options"],
            })
            start_idx += n_options

    def load_dataset(self, use_cache=False):
        T0SingleTaskDataForEncDec.load_dataset(self, use_cache)

    def load_dataloader(self):
        T0SingleTaskDataForEncDec.load_dataloader(self)

    def load_fewshot_data(self, file_path, task, prompt):

        p3_template_name = self.dataset

        # load raw few-shot data
        with open(file_path, "r") as fin:
            data = []
            for idx, line in enumerate(fin.readlines()):
                data.append(json.loads(line.strip("\n")))

        # for perturbation experiments
        if self.config.perturbation_mode == "random_label":
            all_options = list(range(N_OPTIONS[task]))
            for item in data:
                item["label"] = random.choice(all_options) # randomly select 1
        if self.config.perturbation_mode == "wrong_label":
            all_options = list(range(N_OPTIONS[task]))
            for item in data:
                wrong_options = all_options.copy()
                wrong_options.remove(item["label"])
                item["label"] = random.choice(wrong_options)

        if prompt is not None:
            # T0
            prompt = DatasetTemplates(task)[prompt]
            examples = [prompt.apply(item) for item in data]

            _inputs = [result[0] for result in examples]
            _outputs = [result[1] for result in examples]
        else:
            # BIG-Bench (and maybe RAFT)
            _inputs = [dp["input"] for dp in data]
            _outputs = [dp["output"] for dp in data]


        self.logger.info("Tokenizing Input ...")
        tokenized_input = self.tokenizer.batch_encode_plus(_inputs,
                                                        truncation=True,
                                                        max_length=self.config.max_input_length,
                                                        add_special_tokens=False)
        self.logger.info("Tokenizing Output ...")
        tokenized_output = self.tokenizer.batch_encode_plus(_outputs,
                                                    padding='max_length',
                                                    truncation=True,
                                                    max_length=self.config.max_output_length,
                                                    add_special_tokens=self.config.use_eos_for_support_output) # should be true

        input_prefix = self.tokenizer(self.config.input_prefix, add_special_tokens=False)["input_ids"] if self.config.input_prefix else []
        output_prefix = self.tokenizer(self.config.output_prefix, add_special_tokens=False)["input_ids"] if self.config.output_prefix else []

        def concat_input_and_output(_input, _output, input_prefix, output_prefix, pad_token_id):
            if pad_token_id in _input:
                _input = _input[:_input.index(pad_token_id)]
            if pad_token_id in _output:
                _output = _output[:_output.index(pad_token_id)]

            return input_prefix + _input + output_prefix + _output

        def concat_input_or_output(_seq, _prefix, pad_token_id):
            # for perturbation experiments
            if pad_token_id in _seq:
                _seq = _seq[:_seq.index(pad_token_id)]
            return _prefix + _seq       

        if self.config.perturbation_mode in [None, "random_label", "wrong_label"]:
            concat_ids = [
                concat_input_and_output(_input, _output, input_prefix, output_prefix, pad_token_id=self.tokenizer.pad_token_id)
                for _input, _output in zip(tokenized_input["input_ids"], tokenized_output["input_ids"])
            ]
        elif self.config.perturbation_mode == "no_label":
            concat_ids = [
                concat_input_or_output(_input, input_prefix, pad_token_id=self.tokenizer.pad_token_id)
                for _input in tokenized_input["input_ids"]
            ]
        elif self.config.perturbation_mode == "no_input":     
            concat_ids = [
                concat_input_or_output(_output, output_prefix, pad_token_id=self.tokenizer.pad_token_id)
                for _output in tokenized_output["input_ids"]
            ]

        # if self.config.verbose:
        for i in range(3):
            _input = self.tokenizer.decode(concat_ids[i])
            self.logger.info("Example {}".format(i))
            self.logger.info("Text: {}".format(_input))

        max_len = max([len(item) for item in concat_ids])
        concat_ids = self.pad_tokens(concat_ids, max_len=max_len)
        concat_ids = torch.LongTensor(concat_ids)
        concat_attention_mask = concat_ids.ne(self.tokenizer.pad_token_id).long()

        if self.config.varying_shots:
            # at test time we try plugging in different number of shots
            # to see if more shots -> better performance
            concat_ids = concat_ids[:self.config.varying_shots]
            concat_attention_mask = concat_attention_mask[:self.config.varying_shots]
            
        self.support_input_ids = concat_ids
        self.support_attention_mask = concat_attention_mask

        return concat_ids, concat_attention_mask