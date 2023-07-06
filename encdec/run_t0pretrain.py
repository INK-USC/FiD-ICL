import argparse
import os

from transformers import AutoTokenizer

from config import T0PretrainConfig, ParseKwargs
from data_t0pretrain import T0PretrainDataForEncDec
from trainer import Trainer
from utils import seed_everything, init_logger, load_dataset_names, expand_dataset_to_prompts


def run(logger, config):
    # trainer
    trainer = Trainer(config, logger)
    model = trainer.load_model(path=config.init_checkpoint)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    trainer.pad_token_id = tokenizer.pad_token_id
 
    # get prompt data
    datasets = load_dataset_names("t0", "train")
    prompt_identifiers = expand_dataset_to_prompts(datasets)

    train_data = T0PretrainDataForEncDec(
        logger = logger,
        config = config,
        tokenizer = tokenizer,
        datasets = prompt_identifiers,
        data_split = "train",
        is_training = True
    )
    train_data.load_raw_data()
    train_data.load_dataset()

    print(len(train_data.all_input_ids))
    train_data.load_dataloader()

    trainer.do_train(model, train_data, dev_data=None)

if __name__=='__main__':

    parser = argparse.ArgumentParser("Training EncDec Models for Single-task")
    parser.add_argument("-c", "--config_files", default=None)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = T0PretrainConfig(args.config_files, args.kwargs)

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    if not os.path.exists(config.tensorize_dir):
        os.makedirs(config.tensorize_dir)

    seed_everything(config.train_seed)
    logger = init_logger(config)

    logger.info(config.to_json())
    run(logger, config)