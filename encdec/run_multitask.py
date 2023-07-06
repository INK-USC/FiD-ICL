import argparse
import os

from transformers import AutoTokenizer

from config import MultiTaskConfig, ParseKwargs
from data_multitask import MultiTaskDataForEncDec
from trainer import Trainer
from utils import seed_everything, init_logger, load_dataset_names

def run(logger, config):
    # trainer
    trainer = Trainer(config, logger)
    model = trainer.load_model(path=config.init_checkpoint)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    trainer.pad_token_id = tokenizer.pad_token_id

    if config.do_train:
        # data
        datasets = load_dataset_names(config.task, "train")
        train_data = MultiTaskDataForEncDec(
            logger = logger,
            config = config,
            tokenizer = tokenizer,
            datasets = datasets,
            data_split = "train",
            is_training = True
        )
        train_data.load_raw_data()
        train_data.load_dataset()
        train_data.load_dataloader()
        
        # do_train
        trainer.do_train(model, train_data)
    
    if config.do_eval:
        # data
        datasets = load_dataset_names(config.task, "test")
        eval_data = MultiTaskDataForEncDec(
            logger = logger,
            config = config,
            tokenizer = tokenizer,
            datasets = datasets,
            data_split = "test",
            is_training = False
        )
        eval_data.load_raw_data()
        eval_data.load_dataset()
        eval_data.load_dataloader()

        trainer.do_eval(model, eval_data)

if __name__=='__main__':

    parser = argparse.ArgumentParser("Training EncDec Models for Multi-tasking")
    parser.add_argument("-c", "--config_files", default=None)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = MultiTaskConfig(args.config_files, args.kwargs)

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    if not os.path.exists(config.tensorize_dir):
        os.makedirs(config.tensorize_dir)

    seed_everything(config.train_seed)
    logger = init_logger(config)

    logger.info(config.to_json())
    run(logger, config)