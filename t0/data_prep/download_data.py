from t0_config import DATA_SPLITS_SIZES
from datasets import load_dataset
import os
import json
import sys

from multiprocessing import Pool
from functools import partial

def process(key, out_dir):
    print(key)
    splits = DATA_SPLITS_SIZES[key].keys()
    print(splits)
    dataset = load_dataset('bigscience/P3', key)

    if not os.path.exists(os.path.join(out_dir, key)):
        os.makedirs(os.path.join(out_dir, key))
        
    for split in splits:
        d = dataset[split]
        input_ids, target_ids = [], []
        for dp in d:
            input_ids.append(dp["inputs"])
            target_ids.append(dp["targets"])

        out_file = os.path.join(out_dir, key, "{}_{}.json".format(key, split))
        with open(out_file, "w") as fout:
            json.dump([input_ids, target_ids], fout)

    print("=" * 40)

def main():
    out_dir = "../data"
    os.makedirs(out_dir, exist_ok=True)

    keys = DATA_SPLITS_SIZES.keys()

    start = int(sys.argv[1])
    end = int(sys.argv[2])
    
    for i, key in enumerate(keys):
        if i < start:
            continue

        print("Processing {}/{}".format(i, len(keys)))
        process(key, out_dir)

        if i == end:
            break

if __name__ == "__main__":
    main()

        