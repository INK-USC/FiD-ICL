import os
import json

from data_singletask import SingleTaskDataForEncDec

def load_data(base_dir, datasets, split):
    data = []
    for dataset in datasets:
        data_path = os.path.join(base_dir, dataset, "{}_{}.jsonl".format(dataset, split))
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                data.append(dp)
    return data

class T0SingleTaskDataForEncDec(SingleTaskDataForEncDec):
    def get_cache_path(self):
        cache_file = os.path.join(
            self.config.tensorize_dir, 
            "t0_singletask_{}_split={}_k={}_seed={}_lower={}_maxlen={}_mode={}_istraining={}.json".format(
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
