import os
import json
import torch
import numpy as np

from functools import partial
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler, default_collate

from data_t0pretrain import T0PretrainDataForEncDec

class FiDPretrainDataForEncDec(T0PretrainDataForEncDec):
    def load_dataloader(self):
        assert self.is_training # only support training for now 

        n_task = len(self.datasets)
        task_sampler = RandomSampler(range(n_task))
        batch_size = self.config.k + self.config.m # number of support and query examples, the collate function will separate them later
        collate_fn = partial(train_collate_fn, k=self.config.k, m=self.config.m)

        batch_sampler = TaskLevelBatchSampler(
            task_sampler=task_sampler, 
            task_meta_data=self.meta_data, 
            batch_size=batch_size,
            is_training=self.is_training
        )
        self.dataloader = DataLoader(
            self.dataset, 
            batch_sampler=batch_sampler, 
            collate_fn=collate_fn,
        )

    def load_dataset(self):

        def concat_input_and_output(_input, _output, input_prefix, output_prefix, max_input_length, max_output_length):
            return input_prefix + _input[:max_input_length-len(input_prefix)] \
                + output_prefix + _output[:max_output_length-len(output_prefix)]

        # TODO(qinyuany): re-think what's the best way of doing this
        input_prefix = self.tokenizer("# Input #", add_special_tokens=False)["input_ids"]
        output_prefix = self.tokenizer(" # Output #", add_special_tokens=False)["input_ids"]

        concat_ids = [
            concat_input_and_output(_input, _output, input_prefix, output_prefix, self.config.max_input_length, self.config.max_output_length)
            for _input, _output in zip(self.all_input_ids, self.all_target_ids)
        ]
        concat_ids = self.pad_tokens(concat_ids, max_len=self.config.max_input_length+self.config.max_output_length)
        concat_attention_mask = concat_ids.ne(self.tokenizer.pad_token_id).long()
        
        # do not add special token to the in-context examples, so moving this after in-context examples tokenization
        if self.config.fid_special_tokens:
            # 32000 is <extra_id_99>
            self.all_input_ids = [[32000] + item for item in self.all_input_ids]

        input_ids = self.pad_tokens(self.all_input_ids, max_len=self.config.max_input_length)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        decoder_input_ids = self.pad_tokens(self.all_target_ids, max_len=self.config.max_output_length)
        decoder_attention_mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).long()

        self.dataset = TensorDataset(
            input_ids, attention_mask, 
            decoder_input_ids, decoder_attention_mask, 
            concat_ids, concat_attention_mask
        )

def train_collate_fn(data, k, m):
    data = default_collate(data)
    # split a batch into two batches (for support and query) during training
    input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, concat_ids, concat_attention_mask = data
    assert k + m == input_ids.shape[0]
    return concat_ids[:k], concat_attention_mask[:k], \
        input_ids[k:], attention_mask[k:], decoder_input_ids[k:], decoder_attention_mask[k:]

class TaskLevelBatchSampler(BatchSampler):
    def __init__(self, task_sampler, task_meta_data, batch_size, is_training):
        
        self.task_sampler = task_sampler
        self.task_meta_data = task_meta_data
        self.batch_size = batch_size
        self.is_training = is_training

    def __iter__(self):
        sampler_iter = iter(self.task_sampler) # iterator over task indices

        while True:
            try:
                task_idx = next(sampler_iter) # get a task index
                task_name, st, ed = self.task_meta_data[task_idx] # find the task's examples (start position and end position)
                assert self.batch_size <= ed - st
                batch = np.random.choice(range(st, ed), self.batch_size, replace=False).tolist()
                yield batch
            except StopIteration:
                break
    
    def __len__(self):
        return len(self.task_sampler)