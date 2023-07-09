import os
import json
import random
import sys

from t0_config import DATA_SPLITS_SIZES

random.seed(42)

keys = DATA_SPLITS_SIZES.keys()

full_data_dir = "../data"

sampled_data_dir = "../{}".format(sys.argv[1])
upper_limit = int(sys.argv[2])

os.makedirs(sampled_data_dir, exist_ok=True)

for i, key in enumerate(DATA_SPLITS_SIZES.keys()):
    print("Processing {}/660".format(i))
    print(key)

    train_file = os.path.join(full_data_dir, key, "{}_train.json".format(key))
    with open(train_file, "r") as fin:
        input_ids, target_ids = json.load(fin)
    
    print(len(input_ids))

    if not os.path.exists(os.path.join(sampled_data_dir, key)):
        os.makedirs(os.path.join(sampled_data_dir, key))

    out_train_file = os.path.join(sampled_data_dir, key, "{}_train.json".format(key))
    if len(input_ids) > upper_limit:
        input_ids, target_ids = zip(*random.sample(list(zip(input_ids, target_ids)), upper_limit))

    with open(out_train_file, "w") as fout:
        json.dump([input_ids, target_ids], fout)