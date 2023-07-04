import os
import json

from datasets import load_dataset

from t0_config import DATA_SPLITS_SIZES

def expand_dataset_to_prompts(datasets):
    prompt_names = list(DATA_SPLITS_SIZES.keys())
    # select prompts corresponding the the selected datasets
    selected_prompts = filter(
        lambda x: any([x.startswith(item) for item in datasets]) and not x.endswith("score_eval"),
        prompt_names
    )
    selected_prompts = list(selected_prompts)
    return selected_prompts

def process(key, out_dir):
    data = load_dataset("bigscience/P3", key)
    eval_data = data["validation"]
    out_lines = []
    for dp in eval_data:
        out_dict = {
            "task": key,
            "input": dp["inputs_pretokenized"].strip(),
            "output": dp["targets_pretokenized"].strip(),
            "options": dp["answer_choices"] if "answer_choices" in dp else [],
        }
        out_lines.append(out_dict)

    if not os.path.exists(os.path.join(out_dir, key)):
        os.makedirs(os.path.join(out_dir, key))
    with open(os.path.join(out_dir, key, "{}_valid.jsonl".format(key)), "w") as fout:
        for line in out_lines:
            fout.write(json.dumps(line)+"\n")


def main():
    eval_datasets = [
        "super_glue_wsc.fixed",
        "winogrande_winogrande_xl",
        "super_glue_cb",
        "super_glue_rte",
        "anli",
        "super_glue_copa",
        "hellaswag",
        "super_glue_wic"
    ]

    out_dir = "../data_eval"
    filtered_prompt_names = expand_dataset_to_prompts(eval_datasets)
    
    for i, prompt_name in enumerate(filtered_prompt_names):
        print("Processing {}/{}".format(i, len(filtered_prompt_names)))
        process(prompt_name, out_dir)

if __name__ == "__main__":
    main()