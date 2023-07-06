import os

from data_multitask import MultiTaskDataForEncDec

class SingleTaskDataForEncDec(MultiTaskDataForEncDec):
    def __init__(self, logger, config, tokenizer, dataset, data_split, is_training):
        self.dataset = dataset
        super(SingleTaskDataForEncDec, self).__init__(
            logger=logger,
            config=config,
            tokenizer=tokenizer,
            datasets=[dataset],
            data_split=data_split,
            is_training=is_training
        )

    def get_cache_path(self):
        cache_file = os.path.join(
            self.config.tensorize_dir, 
            "singletask_{}_split={}_k={}_seed={}_lower={}_maxlen={}_mode={}_istraining={}.json".format(
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

        
