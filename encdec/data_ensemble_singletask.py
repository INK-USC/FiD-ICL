import os
import json
import torch

from data_t0singletask import T0SingleTaskDataForEncDec, load_data
from data_fid_singletask import FiDSingleTaskDataForEncDec
from data_icl_singletask import eval_collate_fn
from promptsource.templates import DatasetTemplates

from functools import partial
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler, default_collate

class EnsembleSingleTaskDataForEncDec(FiDSingleTaskDataForEncDec):

    def get_cache_path(self):
        cache_file = os.path.join(
            self.config.tensorize_dir, 
            "ensemble_singletask_{}_split={}_k={}_seed={}_lower={}_maxlen={}_mode={}_istraining={}.json".format(
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

        # create one dataloader for one in-context example
        # resulting in a list of dataloaders
        # `do_eval` should loop throught these dataloaders and aggregate the probabilities in the end
        # (trying to make minimal edits to my code)
        self.dataloaders = []

        test_k = self.support_input_ids.shape[0]

        for i in range(test_k):
            support_input_ids = self.support_input_ids[i]
            support_input_ids = support_input_ids[support_input_ids.ne(self.tokenizer.pad_token_id)]
            support_attention_mask = torch.ones_like(support_input_ids, dtype=torch.long)

            collate_fn = partial(
                eval_collate_fn,
                icl_input_ids=support_input_ids,
                icl_attention_mask=support_attention_mask
            )

            dataloader = DataLoader(
                self.dataset, 
                sampler=sampler, 
                batch_size=self.config.eval_batch_size,
                collate_fn=collate_fn
            )

            self.dataloaders.append(dataloader)
