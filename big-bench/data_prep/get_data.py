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
            "output": dp["targets"][0].strip(),
            "options": [item.strip() for item in dp["multiple_choice_targets"]],
        }
        out_lines.append(out_dict)

    if not os.path.exists(os.path.join(out_dir, key)):
        os.makedirs(os.path.join(out_dir, key))

    if split == "validation":
        split = "valid" # for file naming purpose
        # for efficient evaluation i subsample only 1000 examples
        if len(out_lines) > 1000:
            out_lines = random.sample(out_lines, 1000)

    with open(os.path.join(out_dir, key, "{}_{}.jsonl".format(key, split)), "w") as fout:
        for line in out_lines:
            fout.write(json.dumps(line)+"\n")

task_names = [
        "conceptual_combinations",
        "code_line_description",
        "hindu_knowledge",
        "known_unknowns",
        "language_identification",
        "logic_grid_puzzle",
        "logical_deduction",
        "misconceptions",
        "movie_dialog_same_or_different",
        "novel_concepts",
        "strategyqa",
        "formal_fallacies_syllogisms_negation",
        "vitaminc_fact_verification",
        "winowhy"
	]
print(task_names)

out_dir = "../data"
os.makedirs(out_dir, exist_ok=True)

for task_name in task_names:
    # run `process_strategyqa.py` for strategyqa
    if task_name == "strategyqa":
        continue
    process(task_name, "train", out_dir)
    process(task_name, "validation", out_dir)
    

