import argparse
import os
import torch
import pandas as pd

from transformers import AutoTokenizer

from config import SingleTaskConfig, ParseKwargs
from data_t0singletask import T0SingleTaskDataForEncDec
from trainer import Trainer
from utils import seed_everything, init_logger, load_dataset_names, expand_dataset_to_prompts, map_prompt_name_to_task_name
from task_configs.t0_config import split_infos

def run(logger, config):
    # trainer
    trainer = Trainer(config, logger)
    model = trainer.load_model(path=config.init_checkpoint)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    trainer.pad_token_id = tokenizer.pad_token_id

    # get prompt data
    prompt_identifiers = load_dataset_names(config.task, "eval")
    datasets = load_dataset_names(config.task, "eval_datasets")
    # prompt_identifiers = ["super_glue_wsc.fixed_GPT_3_Style"]

    df = pd.DataFrame(columns=["task_name", "prompt_name", "performance", "eval_mode"])
    results_file = os.path.join(config.out_dir, "results_{}.csv".format(config.task))

    for i, prompt in enumerate(prompt_identifiers):
        logger.info("Evaluation {}/{}".format(i, len(prompt_identifiers)))

        # map prompt name to the original task name (for better result aggregation)
        task_name = map_prompt_name_to_task_name(prompt, datasets)

        # figure out the evaluation mode (generation or rank classification)
        # story_cloze is not in that `t0_config.py` list
        config.eval_mode = "rank_classification" if task_name == "story_cloze" or "answer_choices" in split_infos[prompt]["features"] else "generation"

        # data
        eval_data = T0SingleTaskDataForEncDec(
            logger = logger,
            config = config,
            tokenizer = tokenizer,
            dataset = prompt,
            data_split = "valid",
            is_training = False
        )
        eval_data.load_raw_data()
        eval_data.load_dataset(use_cache=False)
        eval_data.load_dataloader()

        metric, test_perf = trainer.do_eval(model, eval_data)

        df.loc[len(df.index)] = [task_name, prompt, test_perf, config.eval_mode]
        
        # save after each dataset is evaluated
        df.to_csv(results_file)

    # aggreate results and get mean/median
    agg_results_file = os.path.join(config.out_dir, "results_agg_{}.csv".format(config.task))

    mean = df.groupby("task_name").mean()
    median = df.groupby("task_name").median()
    mean = mean.rename(columns={"performance": "mean"})
    mean["median"] = median["performance"]
    mean.to_csv(agg_results_file)

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