import os
import json
import torch

from data_t0singletask import T0SingleTaskDataForEncDec, load_data
from data_fid_singletask import FiDSingleTaskDataForEncDec
from promptsource.templates import DatasetTemplates

from functools import partial
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler, default_collate

class ICLSingleTaskDataForEncDec(FiDSingleTaskDataForEncDec):

    def get_cache_path(self):
        cache_file = os.path.join(
            self.config.tensorize_dir, 
            "icl_singletask_{}_split={}_k={}_seed={}_lower={}_maxlen={}_mode={}_istraining={}.json".format(
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

    def load_dataloader(self):
        sampler = SequentialSampler(self.dataset)

        # prepend the in-context learning examples in the collate_fn
        # (trying to make minimal edits to my code)
        collate_fn = partial(
            eval_collate_fn,
            icl_input_ids=self.support_input_ids,
            icl_attention_mask=self.support_attention_mask
        )

        self.dataloader = DataLoader(
            self.dataset, 
            sampler=sampler, 
            batch_size=self.config.eval_batch_size,
            collate_fn=collate_fn
        )

    def load_fewshot_data(self, file_path, task, prompt):

        # reuse the fid loader, but flatten the input_ids into a long sequence at the end
        # this function handles `config.varying_shots` too...
        concat_ids, concat_attention_mask = FiDSingleTaskDataForEncDec.load_fewshot_data(self, file_path, task, prompt)

        concat_ids = concat_ids.flatten()
        concat_ids = concat_ids[concat_ids.ne(self.tokenizer.pad_token_id)]
        concat_attention_mask = torch.ones(concat_ids.shape[0], dtype=torch.long)

        # if self.config.verbose:
        _input = self.tokenizer.decode(concat_ids)
        self.logger.info("In-context Examples: {}".format(_input))

        self.support_input_ids = concat_ids
        self.support_attention_mask = concat_attention_mask

        return concat_ids, concat_attention_mask

def eval_collate_fn(data, icl_input_ids, icl_attention_mask):
    data = default_collate(data)
    input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = data

    bsz = input_ids.shape[0]
    icl_input_ids = icl_input_ids.unsqueeze(0).repeat(bsz, 1)
    icl_attention_mask = icl_attention_mask.unsqueeze(0).repeat(bsz, 1) # should be all ones

    new_input_ids = torch.cat([icl_input_ids, input_ids], dim=1)
    new_attention_mask = torch.cat([icl_attention_mask, attention_mask], dim=1)

    return new_input_ids, new_attention_mask, decoder_input_ids, decoder_attention_mask
    
