import argparse
import os
import torch
import pandas as pd

from transformers import AutoTokenizer

from config import SingleTaskConfig, ParseKwargs
from data_t0singletask import T0SingleTaskDataForEncDec
from data_fid_singletask import FiDSingleTaskDataForEncDec
from data_icl_singletask import ICLSingleTaskDataForEncDec
from data_ensemble_singletask import EnsembleSingleTaskDataForEncDec
from data_fid_pair_singletask import FiDPairSingleTaskDataForEncDec
from trainer import Trainer
from trainer_fid import FiDTrainer
from trainer_ensemble import EnsembleTrainer
from trainer_fid_pair import FiDPairTrainer
from trainer_fid_ensemble import FiDEnsembleTrainer
from utils import seed_everything, init_logger, load_dataset_names, expand_dataset_to_prompts, map_prompt_name_to_task_name
from task_configs.t0_config import split_infos

def eval_one_checkpoint_vanilla(logger, config, step, checkpoint):
    # trainer
    trainer = Trainer(config, logger)
    model = trainer.load_model(path=checkpoint)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    trainer.tokenizer = tokenizer
    trainer.pad_token_id = tokenizer.pad_token_id

    # get prompt data
    task_names = load_dataset_names("big_bench", "eval")

    df = pd.DataFrame(columns=["task_name", "performance", "eval_mode"])
    results_file = os.path.join(config.out_dir, "results_{}.csv".format(step))

    for i, task_name in enumerate(task_names):
        logger.info("Evaluation {}/{}".format(i, len(task_names)))

        config.eval_mode = "rank_classification"

        # data
        eval_data = T0SingleTaskDataForEncDec(
            logger = logger,
            config = config,
            tokenizer = tokenizer,
            dataset = task_name,
            data_split = "valid",
            is_training = False
        )
        eval_data.load_raw_data()
        eval_data.load_dataset(use_cache=False)
        eval_data.load_dataloader()

        metric, test_perf = trainer.do_eval(model, eval_data)

        df.loc[len(df.index)] = [task_name, test_perf, config.eval_mode]
        
        # save after each dataset is evaluated
        df.to_csv(results_file)

    # show avg
    avg = df["performance"].mean()
    df.loc[len(df.index)] = ["AVG", avg, config.eval_mode]
    df.to_csv(results_file)
    
    return avg

def eval_one_checkpoint_fid(logger, config, step, checkpoint):
    # trainer
    trainer = FiDTrainer(config, logger)
    model = trainer.load_model(path=checkpoint)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    trainer.tokenizer = tokenizer
    trainer.pad_token_id = tokenizer.pad_token_id

    task_names = load_dataset_names("big_bench", "eval")
    df = pd.DataFrame(columns=["task_name", "performance", "eval_mode"])
    results_file = os.path.join(config.out_dir, "results_{}.csv".format(step))


    for i, task_name in enumerate(task_names):
        logger.info("Evaluation {}/{}".format(i, len(task_names)))

        config.eval_mode = "rank_classification"

        # data
        eval_data = FiDSingleTaskDataForEncDec(
            logger = logger,
            config = config,
            tokenizer = tokenizer,
            dataset = task_name,
            data_split = "valid",
            is_training = False
        )
        eval_data.load_raw_data()
        eval_data.load_dataset(use_cache=False)
        eval_data.load_dataloader()

        filename = "../big-bench/data/{}/{}_train.jsonl".format(task_name, task_name)
        concat_ids, concat_attention_mask = eval_data.load_fewshot_data(filename, task_name, prompt=None)

        metric, test_perf = trainer.do_eval(model, eval_data)

        df.loc[len(df.index)] = [task_name, test_perf, config.eval_mode]
        
        # save after each dataset is evaluated
        df.to_csv(results_file)

    # show avg
    avg = df["performance"].mean()
    df.loc[len(df.index)] = ["AVG", avg, config.eval_mode]
    df.to_csv(results_file)
    
    return avg

def eval_one_checkpoint_fid_ensemble(logger, config, step, checkpoint):
    trainer = FiDEnsembleTrainer(config, logger)
    model = trainer.load_model(path=checkpoint)    

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    trainer.tokenizer = tokenizer
    trainer.pad_token_id = tokenizer.pad_token_id

    # get prompt data
    task_names = load_dataset_names("big_bench", "eval")

    df = pd.DataFrame(columns=["task_name", "performance", "eval_mode"])
    results_file = os.path.join(config.out_dir, "results_{}.csv".format(step))

    for i, task_name in enumerate(task_names):
        logger.info("Evaluation {}/{}".format(i, len(task_names)))

        config.eval_mode = "rank_classification"

        # data (re-using FiD dataloader: the only diff is how the data goes through the eval loop)
        eval_data = FiDSingleTaskDataForEncDec(
            logger = logger,
            config = config,
            tokenizer = tokenizer,
            dataset = task_name,
            data_split = "valid",
            is_training = False
        )
        eval_data.load_raw_data()
        eval_data.load_dataset(use_cache=False)
        eval_data.load_dataloader()

        filename = "../big-bench/data/{}/{}_train.jsonl".format(task_name, task_name)
        concat_ids, concat_attention_mask = eval_data.load_fewshot_data(filename, task_name, prompt=None)

        metric, test_perf = trainer.do_eval(model, eval_data)

        df.loc[len(df.index)] = [task_name, test_perf, config.eval_mode]
        
        # save after each dataset is evaluated
        df.to_csv(results_file)

    # show avg
    avg = df["performance"].mean()
    df.loc[len(df.index)] = ["AVG", avg, config.eval_mode]
    df.to_csv(results_file)
    
    return avg

def eval_one_checkpoint_fid_pair(logger, config, step, checkpoint):
    # trainer
    trainer = FiDPairTrainer(config, logger)
    model = trainer.load_model(path=checkpoint)    

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    trainer.tokenizer = tokenizer
    trainer.pad_token_id = tokenizer.pad_token_id

    # get prompt data
    task_names = load_dataset_names("big_bench", "eval")

    df = pd.DataFrame(columns=["task_name", "performance", "eval_mode"])
    results_file = os.path.join(config.out_dir, "results_{}.csv".format(step))

    for i, task_name in enumerate(task_names):
        logger.info("Evaluation {}/{}".format(i, len(task_names)))

        config.eval_mode = "rank_classification"

        # data
        eval_data = FiDPairSingleTaskDataForEncDec(
            logger = logger,
            config = config,
            tokenizer = tokenizer,
            dataset = task_name,
            data_split = "valid",
            is_training = False
        )
        eval_data.load_raw_data()
        eval_data.load_dataset(use_cache=False)

        filename = "../big-bench/data/{}/{}_train.jsonl".format(task_name, task_name)
        concat_ids, concat_attention_mask = eval_data.load_fewshot_data(filename, task_name, prompt=None)
        eval_data.load_dataloader()

        metric, test_perf = trainer.do_eval(model, eval_data)

        df.loc[len(df.index)] = [task_name, test_perf, config.eval_mode]
        
        # save after each dataset is evaluated
        df.to_csv(results_file)

    # show avg
    avg = df["performance"].mean()
    df.loc[len(df.index)] = ["AVG", avg, config.eval_mode]
    df.to_csv(results_file)
    
    return avg

def eval_one_checkpoint_icl(logger, config, step, checkpoint):
    # trainer
    trainer = Trainer(config, logger)
    model = trainer.load_model(path=checkpoint)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    trainer.tokenizer = tokenizer
    trainer.pad_token_id = tokenizer.pad_token_id

    # get prompt data
    task_names = load_dataset_names("big_bench", "eval")

    df = pd.DataFrame(columns=["task_name", "performance", "eval_mode"])
    results_file = os.path.join(config.out_dir, "results_{}.csv".format(step))

    for i, task_name in enumerate(task_names):
        logger.info("Evaluation {}/{}".format(i, len(task_names)))

        config.eval_mode = "rank_classification"

        # data
        eval_data = ICLSingleTaskDataForEncDec(
            logger = logger,
            config = config,
            tokenizer = tokenizer,
            dataset = task_name,
            data_split = "valid",
            is_training = False
        )
        eval_data.load_raw_data()
        eval_data.load_dataset(use_cache=False)

        filename = "../big-bench/data/{}/{}_train.jsonl".format(task_name, task_name)
        concat_ids, concat_attention_mask = eval_data.load_fewshot_data(filename, task_name, prompt=None)
        eval_data.load_dataloader() # update the collate_fn and thus use the new set of few-shots

        metric, test_perf = trainer.do_eval(model, eval_data)

        df.loc[len(df.index)] = [task_name, test_perf, config.eval_mode]
        
        # save after each dataset is evaluated
        df.to_csv(results_file)

    # show avg
    avg = df["performance"].mean()
    df.loc[len(df.index)] = ["AVG", avg, config.eval_mode]
    df.to_csv(results_file)
    
    return avg    

def eval_one_checkpoint_ensemble(logger, config, step, checkpoint):
    # trainer
    trainer = EnsembleTrainer(config, logger)
    model = trainer.load_model(path=checkpoint)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    trainer.tokenizer = tokenizer
    trainer.pad_token_id = tokenizer.pad_token_id

    # get prompt data
    task_names = load_dataset_names("big_bench", "eval")

    df = pd.DataFrame(columns=["task_name", "performance", "eval_mode"])
    results_file = os.path.join(config.out_dir, "results_{}.csv".format(step))

    for i, task_name in enumerate(task_names):
        logger.info("Evaluation {}/{}".format(i, len(task_names)))

        eval_data = EnsembleSingleTaskDataForEncDec(
            logger = logger,
            config = config,
            tokenizer = tokenizer,
            dataset = task_name,
            data_split = "valid",
            is_training = False
        )
        eval_data.load_raw_data()
        eval_data.load_dataset()

        filename = "../big-bench/data/{}/{}_train.jsonl".format(task_name, task_name)
        concat_ids, concat_attention_mask = eval_data.load_fewshot_data(filename, task_name, prompt=None)
        eval_data.load_dataloader() # update the collate_fn and thus use the new set of few-shots

        metric, test_perf = trainer.do_eval(model, eval_data)

        df.loc[len(df.index)] = [task_name, test_perf, config.eval_mode]
                
        # save after each dataset is evaluated
        df.to_csv(results_file)

    # show avg
    avg = df["performance"].mean()
    df.loc[len(df.index)] = ["AVG", avg, config.eval_mode]
    df.to_csv(results_file)
    
    return avg  
            

def run(logger, config):
    if config.model_mode == "vanilla":
        eval_one_checkpoint = eval_one_checkpoint_vanilla
    elif config.model_mode == "fid":
        eval_one_checkpoint = eval_one_checkpoint_fid
    elif config.model_mode == "icl":
        eval_one_checkpoint = eval_one_checkpoint_icl
    elif config.model_mode == "ensemble":
        eval_one_checkpoint = eval_one_checkpoint_ensemble
    elif config.model_mode == "fid_pair":
        eval_one_checkpoint = eval_one_checkpoint_fid_pair
    elif config.model_mode == "fid_ensemble":
        eval_one_checkpoint = eval_one_checkpoint_fid_ensemble

    if not config.do_eval_all:
        eval_one_checkpoint(logger, config, "na", config.init_checkpoint)
    else:
        checkpoints = filter(lambda x: x.startswith("model-") and x[6:-3].isdigit(), os.listdir(config.init_checkpoint))
        checkpoints = sorted(list(checkpoints), key=lambda x: int(x[6:-3]))

        result_dict = {}
        for checkpoint in checkpoints:
            logger.info("Evaluating Checkpoint {}".format(checkpoint))
            step = int(checkpoint[6:-3])
            checkpoint_file = os.path.join(config.init_checkpoint, checkpoint)
            avg_performance = eval_one_checkpoint(logger, config, step, checkpoint_file)
            result_dict[checkpoint] = avg_performance

        logger.info("Performance at all checkpoints: {}".format(result_dict))
        best = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        logger.info("Best checkpoint: {}".format(best[0]))


if __name__=='__main__':

    parser = argparse.ArgumentParser("Training EncDec Models for Big-Bench Evaluation")
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