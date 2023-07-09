import argparse
import os

from transformers import AutoTokenizer

from config import SingleTaskConfig, ParseKwargs
from data_singletask import SingleTaskDataForEncDec
from trainer import Trainer
from utils import seed_everything, init_logger, load_dataset_names

def run(logger, config):
    # trainer
    trainer = Trainer(config, logger)
    model = trainer.load_model(path=config.init_checkpoint)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    trainer.pad_token_id = tokenizer.pad_token_id

    # avoid errors when return
    best_dev_perf, test_perf, metric = -1, -1, None

    if config.do_train:
        # data
        train_data = SingleTaskDataForEncDec(
            logger = logger,
            config = config,
            tokenizer = tokenizer,
            dataset = config.dataset,
            data_split = "train",
            is_training = True
        )
        train_data.load_raw_data()
        train_data.load_dataset()
        train_data.load_dataloader()

        if config.do_valid:
            dev_data = SingleTaskDataForEncDec(
                logger = logger,
                config = config,
                tokenizer = tokenizer,
                dataset = config.dataset,
                data_split = "dev",
                is_training = False
            )
            dev_data.load_raw_data()
            dev_data.load_dataset()
            dev_data.load_dataloader()
        else:
            dev_data = None
        
        # do_train
        best_dev_perf = trainer.do_train(model, train_data, dev_data)
    
    if config.do_eval:
        if config.do_train:
            if config.do_valid:
                model_path = os.path.join(config.out_dir, "model-best.pt")
            else:
                model_path = os.path.join(config.out_dir, "model-last.pt")
            model = trainer.load_model(path=model_path)
        # data
        eval_data = SingleTaskDataForEncDec(
            logger = logger,
            config = config,
            tokenizer = tokenizer,
            dataset = config.dataset,
            data_split = "test",
            is_training = False
        )
        eval_data.load_raw_data()
        eval_data.load_dataset()
        eval_data.load_dataloader()

        metric, test_perf = trainer.do_eval(model, eval_data)
        
        # temporary solution: write out to result.txt
        with open(os.path.join(config.out_dir, "result.txt"), "w") as fout:
            fout.write("{:.8f}".format(test_perf))

    return best_dev_perf, test_perf, metric

if __name__=='__main__':

    parser = argparse.ArgumentParser("Training EncDec Models for Single-task")
    parser.add_argument("-c", "--config_files", default=None)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = SingleTaskConfig(args.config_files, args.kwargs)

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    if not os.path.exists(config.tensorize_dir):
        os.makedirs(config.tensorize_dir)

    seed_everything(config.train_seed)
    logger = init_logger(config)

    logger.info(config.to_json())
    run(logger, config)