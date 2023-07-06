import argparse
import os
import time

import pandas as pd
from transformers import AutoTokenizer

from config import FiDConfig, ParseKwargs
from data_ensemble_singletask import EnsembleSingleTaskDataForEncDec
from trainer_ensemble import EnsembleTrainer
from utils import seed_everything, init_logger, load_dataset_names, expand_dataset_to_prompts, get_caconical_name
from task_configs.fid_eval import FID_METADATA

def run(logger, config):
    # trainer
    trainer = EnsembleTrainer(config, logger)
    model = trainer.load_model(path=config.init_checkpoint)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    trainer.tokenizer = tokenizer
    trainer.pad_token_id = tokenizer.pad_token_id

    df = pd.DataFrame(columns=["task_name", "prompt_name", "seed", "performance"])
    results_file = os.path.join(config.out_dir, "results_{}.csv".format(config.task))

    time_sum = 0
    # for dataset in FID_METADATA.keys():
    for dataset in ["story_cloze/2016"]:
    # for dataset in ["super_glue/wsc.fixed", "winogrande/winogrande_xl", "super_glue/cb", "super_glue/rte", "super_glue/copa", "super_glue/wic", "story_cloze/2016"]:
        for prompt in FID_METADATA[dataset][:1]:
            # get canonical name
            cname = get_caconical_name(dataset, prompt)

            eval_data = EnsembleSingleTaskDataForEncDec(
                logger = logger,
                config = config,
                tokenizer = tokenizer,
                dataset = cname,
                data_split = "valid",
                is_training = False
            )
            eval_data.load_raw_data()
            eval_data.load_dataset()

            # get few-shot data
            for seed in [0, 1, 32, 42, 1024]:
                start_time = time.time()
                clean_dataset = dataset.replace("/", "_")
                if clean_dataset.endswith("_2016"):
                    clean_dataset = clean_dataset[:-5]
                filename = "../t0/data_fewshot/{}/16_shot/{}_seed.jsonl".format(clean_dataset, seed)
                if not os.path.exists(filename):
                    print(filename)
            
                clean_dataset = "anli" if dataset.startswith("anli") else dataset
                concat_ids, concat_attention_mask = eval_data.load_fewshot_data(filename, clean_dataset, prompt)
                eval_data.load_dataloader() # update the collate_fn and thus use the new set of few-shots

                metric, test_perf = trainer.do_eval(model, eval_data)

                df.loc[len(df.index)] = [dataset, prompt, seed, test_perf]
                
                # # save after each dataset is evaluated
                df.to_csv(results_file)

                end_time = time.time()
                logger.info("[{}-shot] time elapsed: {}".format(config.varying_shots, end_time - start_time))
                time_sum += end_time - start_time

    logger.info("[{}-shot] throughput: {}".format(config.varying_shots, (5.0 * 277) / time_sum))
if __name__=='__main__':

    parser = argparse.ArgumentParser("Evaluating Ensemble Models")
    parser.add_argument("-c", "--config_files", default=None)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = FiDConfig(args.config_files, args.kwargs)

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    if not os.path.exists(config.tensorize_dir):
        os.makedirs(config.tensorize_dir)

    seed_everything(config.train_seed)
    logger = init_logger(config)

    logger.info(config.to_json())
    run(logger, config)