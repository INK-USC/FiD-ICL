import os
import json
import torch
import numpy as np

from functools import partial
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler, default_collate

from data_fid import FiDPretrainDataForEncDec

# only need to change the dataloader
class ICLPretrainDataForEncDec(FiDPretrainDataForEncDec):
    def load_dataloader(self):
        assert self.is_training # only support training for now 

        n_task = len(self.datasets)
        task_sampler = RandomSampler(range(n_task))
        batch_size = self.config.k + self.config.m # number of support and query examples, the collate function will separate them later
        collate_fn = partial(
            train_collate_fn, 
            test_k=self.config.test_k, batch_size=self.config.batch_size,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.config.test_k * (self.config.max_input_length + self.config.max_output_length) + self.config.max_input_length
        )

        batch_sampler = ICLBatchSampler(
            task_sampler=task_sampler, 
            task_meta_data=self.meta_data, 
            batch_size=self.config.test_k + 1, # `test_k` in-context examples and 1 inference example
            task_batch_size=self.config.batch_size,
            is_training=self.is_training
        )
        self.dataloader = DataLoader(
            self.dataset, 
            batch_sampler=batch_sampler, 
            collate_fn=collate_fn,
        )

def train_collate_fn(data, test_k, batch_size, pad_token_id, max_length):

    def concat_two_tensors(a, b, pad_token_id):
        # concat two tensors while removing the pad_tokens
        cat = torch.cat([a, b])
        return cat[cat.ne(pad_token_id)]

    data = default_collate(data)
    # split a batch into two batches (for support and query) during training
    input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, concat_ids, concat_attention_mask = [
        item.view(batch_size, test_k+1, -1) for item in data
    ]

    new_input_ids = torch.zeros((batch_size, max_length), dtype=input_ids.dtype)
    new_attention_mask = torch.zeros((batch_size, max_length), dtype=attention_mask.dtype)

    for i in range(batch_size):
        new_seq = concat_two_tensors(concat_ids[i, :-1, :].flatten(), input_ids[i, -1, :], pad_token_id)
        new_seqlen = len(new_seq)
        new_input_ids[i][:new_seqlen] = new_seq
        new_attention_mask[i][:new_seqlen] = 1

    decoder_input_ids = decoder_input_ids[:, -1, :] # bsz * seqlen
    decoder_attention_mask = decoder_attention_mask[:, -1, :] # bsz * seqlen

    return new_input_ids, new_attention_mask, decoder_input_ids, decoder_attention_mask

class ICLBatchSampler(BatchSampler):
    def __init__(self, task_sampler, task_meta_data, batch_size, task_batch_size, is_training):
        
        self.task_sampler = task_sampler
        self.task_meta_data = task_meta_data
        self.batch_size = batch_size
        self.task_batch_size = task_batch_size
        self.is_training = is_training
        self.drop_last = True

    def __iter__(self):
        sampler_iter = iter(self.task_sampler) # iterator over task indices

        while True:
            try:
                # randomly sample tasks
                all_task_ids = [next(sampler_iter) for _ in range(self.task_batch_size)]
                batch = []
                for task_id in all_task_ids:
                    task_idx = next(sampler_iter) # get a task index
                    task_name, st, ed = self.task_meta_data[task_idx] # find the task's examples (start position and end position)
                    assert self.batch_size <= ed - st
                    # randomly sample instances within the selected task
                    instances = np.random.choice(range(st, ed), self.batch_size, replace=False).tolist()
                    batch.extend(instances)

                # batch is a 1d matrix (num_tasks x num_shots)
                # need to reshape back to 2d later in the collator
                yield batch
            except StopIteration:
                break
    
    def __len__(self):
        return len(self.task_sampler) // self.task_batch_size