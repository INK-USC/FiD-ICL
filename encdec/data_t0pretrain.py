import os
import json
import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from data_multitask import MultiTaskDataForEncDec

class T0PretrainDataForEncDec(MultiTaskDataForEncDec):
    # no need to modify init

    def __len__(self):
        return len(self.all_input_ids)

    def load_raw_data(self):
        self.all_input_ids, self.all_target_ids = [], []
        self.meta_data = [] # marks the start and end index of each dataset

        for dataset in self.datasets:
            data_file = os.path.join(
                self.config.data_dir, 
                dataset, 
                "{}_{}.json".format(dataset, self.data_split)
            )
            with open(data_file, "r") as fin:
                input_ids, target_ids = json.load(fin)
                st = len(self.all_input_ids)
                self.all_input_ids.extend(input_ids)
                self.all_target_ids.extend(target_ids)
                ed = len(self.all_input_ids)
                self.meta_data.append((dataset, st, ed))

        if self.config.verbose:
            self.logger.info("Printing 3 examples ...")
            for i in range(3):
                _input = self.tokenizer.decode(self.all_input_ids[i])
                _output = self.tokenizer.decode(self.all_target_ids[i])
                self.logger.info("Example {}".format(i))
                self.logger.info("Input: {}".format(_input))
                self.logger.info("Output: {}".format(_output))

    def pad_tokens(self, lst, max_len, pad_id=None):
        # everything is padding token in the beginning
        if pad_id is None:
            pad_id = self.tokenizer.pad_token_id
            
        tensor = torch.ones(len(lst), max_len, dtype=torch.long) * pad_id
        # then fill each example into this big tensor
        for i, item in enumerate(lst):
            if len(item) > max_len:
                tensor[i, :] = torch.LongTensor(item[:max_len])
            else:
                tensor[i, :len(item)] = torch.LongTensor(item)
        return tensor  

    def load_dataset(self):

        input_ids = self.pad_tokens(self.all_input_ids, max_len=self.config.max_input_length)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        decoder_input_ids = self.pad_tokens(self.all_target_ids, max_len=self.config.max_output_length, pad_id=-100)
        decoder_attention_mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).long()
        
        self.dataset = TensorDataset(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)

