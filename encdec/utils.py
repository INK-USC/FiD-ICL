import logging
import os
import json
import random
import numpy as np
import torch
from task_configs.t0_config import DATA_SPLITS_SIZES

# copied from metaicl

def load_data(base_dir, datasets, split, k, seed):
    data = []
    for dataset in datasets:
        data_path = os.path.join(base_dir, dataset,
                                "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                data.append(dp)
    return data

def load_dataset_names(task, split):
    with open(os.path.join("task_configs", task+".json"), "r") as f:
        config = json.load(f)
    datasets = config[split]
    return datasets

def expand_dataset_to_prompts(datasets):
    prompt_names = list(DATA_SPLITS_SIZES.keys())
    # select prompts corresponding the the selected datasets
    selected_prompts = filter(
        lambda x: any([x.startswith(item) for item in datasets]),
        prompt_names
    )
    selected_prompts = list(selected_prompts)
    return selected_prompts

def map_prompt_name_to_task_name(prompt_name, task_names):
    elements = prompt_name.split("_")

    # special handling for anli_r1/r2/r3 (r1/r2/r3 appears at the end of the prompt name)
    if elements[0] == "anli":
        return "{}_{}".format(elements[0], elements[-1])
    for task_name in task_names:
        if prompt_name.startswith(task_name):
            return task_name
    raise NotImplementedError()

def get_caconical_name(dataset, prompt):
    clean_prompt_name = prompt.replace("-", " ").replace(",", " ").replace("/", " ").replace("…", " ").replace("?", " ").replace("  ", "_").replace(" ", "_")
    if "anli" in dataset:
        c_prompt = "anli_" + clean_prompt_name + "_" + dataset[-2:]
    elif "story_cloze" in dataset:
        c_prompt = "story_cloze_" + clean_prompt_name
    else:
        c_prompt = dataset.replace("/", "_") + "_" + clean_prompt_name
    return c_prompt
    
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def init_logger(config):
    handlers = [logging.StreamHandler()]
    if config.log_file is not None:
        filename = os.path.join(config.out_dir, config.log_file)
        handlers.append(logging.FileHandler(filename))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        force=True,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    return logger

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

def trim_batch_3d(
    hidden_states,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by 0 in attention_mask"""
    # used in FiD-Pair
    keep_column_mask = attention_mask.ne(0).any(dim=0)
    return hidden_states[:, keep_column_mask, :], attention_mask[:, keep_column_mask]

def label_smoothed_nll_loss(lprobs, target, epsilon=0.1, ignore_index=-100, average="instance"):
    """From fairseq"""

    target = target.unsqueeze(-1) # add an extra dimension for gathering, [bsz, seqlen, 1]
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    pad_mask = target.eq(ignore_index)
    nll_loss.masked_fill_(pad_mask, 0.0)
    smooth_loss.masked_fill_(pad_mask, 0.0)

    eps_i = epsilon / lprobs.size(-1)

    if average == "sum":
        # return the sum of loss over all tokens in this batch
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    if average == "average":
        # return the average loss for all tokens in this batch (should be identical to output.loss)
        n_tokens = target.ne(ignore_index).sum()
        nll_loss = nll_loss.sum() / n_tokens
        smooth_loss = smooth_loss.sum() / n_tokens
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    if average == "instance_sum":
        # for each instance, compute the sum of loss over all tokens
        # return a 1d tensor, representing loss for each instance
        nll_loss = nll_loss.squeeze(-1).sum(-1)
        smooth_loss = smooth_loss.squeeze(-1).sum(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss
        
    if average == "instance":
        # for each instance, compute the per-token average loss
        # return a 1d tensor, representing loss for each instance
        token_per_instance = target.shape[-2] - pad_mask.sum(1)
        token_per_instance = token_per_instance.squeeze(-1)
        nll_loss = nll_loss.squeeze(-1).sum(-1) / token_per_instance
        smooth_loss = smooth_loss.squeeze(-1).sum(-1) / token_per_instance
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    if average is None:
        # return non-normalized, per-token loss
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss


