import json
import os
from datasets import load_dataset
import random

random.seed(42)

def process(key, split, out_dir):
    data = load_dataset("bigbench", key)
    eval_data = data[split]
    out_lines = []
    for dp in eval_data:
        if len(dp["targets"]) > 1:
            breakpoint()

        out_dict = {
            "task": key,
            "input": dp["inputs"].strip(),
            "output": dp["targets"][0].split(".")[0],
            "options": [item.strip() for item in dp["multiple_choice_targets"]],
        }
        out_lines.append(out_dict)

    if not os.path.exists(os.path.join(out_dir, key)):
        os.makedirs(os.path.join(out_dir, key))

    if split == "validation":
        split = "valid" # for file naming purpose
        if len(out_lines) > 1000:
            out_lines = random.sample(out_lines, 1000)
    with open(os.path.join(out_dir, key, "{}_{}.jsonl".format(key, split)), "w") as fout:
        for line in out_lines:
            fout.write(json.dumps(line)+"\n")

out_dir = "../data"

process("strategyqa", "train", out_dir)
process("strategyqa", "validation", out_dir)


