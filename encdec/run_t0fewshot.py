import argparse
import os
import pandas as pd

from transformers import AutoTokenizer

from config import SingleTaskConfig, ParseKwargs
from data_t0singletask import T0SingleTaskDataForEncDec
from data_t0fewshot import T0FewshotDataForEncDec
from trainer import Trainer
from utils import seed_everything, init_logger, load_dataset_names, get_caconical_name
from task_configs.fid_eval import FID_METADATA

def run(logger, config, dataset, prompt, seed):
    # trainer
    trainer = Trainer(config, logger)
    model = trainer.load_model(path=config.init_checkpoint)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    trainer.pad_token_id = tokenizer.pad_token_id

    # avoid errors when return
    best_dev_perf, test_perf, metric = -1, -1, None

    # data
    train_data = T0FewshotDataForEncDec(
        logger = logger,
        config = config,
        tokenizer = tokenizer,
        dataset = dataset,
        data_split = "train",
        is_training = True
    )
    train_data.load_raw_data(dataset, prompt, seed)
    train_data.load_dataset()
    train_data.load_dataloader()


    # do_train
    best_dev_perf = trainer.do_train(model, train_data, dev_data=None)
 
    # just eval the last model
    # data
    cname = get_caconical_name(dataset, prompt)
    eval_data = T0SingleTaskDataForEncDec(
        logger = logger,
        config = config,
        tokenizer = tokenizer,
        dataset = cname,
        data_split = "valid",
        is_training = False
    )
    eval_data.load_raw_data()
    eval_data.load_dataset()
    eval_data.load_dataloader()

    metric, test_perf = trainer.do_eval(model, eval_data)

    return best_dev_perf, test_perf, metric

def main(logger, config):
    df = pd.DataFrame(columns=["task_name", "prompt_name", "seed", "performance", "metric"])
    results_file = os.path.join(config.out_dir, "results_{}.csv".format(config.task))

    # for dataset in FID_METADATA.keys():
    # for dataset in ["super_glue/wsc.fixed", "winogrande/winogrande_xl", "super_glue/cb", "super_glue/rte", "super_glue/copa", "super_glue/wic", "story_cloze/2016"]:
    for dataset in ["super_glue/rte", "story_cloze/2016"]:
        for prompt in FID_METADATA[dataset][:1]:
            for seed in [0, 1, 32, 42, 1024]:
                logger.info("Running {} | {} | {} ".format(dataset, prompt, seed))
                _, test_perf, metric = run(logger, config, dataset, prompt, seed)

                df.loc[len(df.index)] = [dataset, prompt, seed, test_perf, metric]
                df.to_csv(results_file)

if __name__=='__main__':

    parser = argparse.ArgumentParser("Training EncDec Models for T0 Few-shot FT")
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
    main(logger, config)