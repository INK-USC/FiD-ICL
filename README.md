## FiD-ICL: A Fusion-in-Decoder Approach for Efficient In-Context Learning


This repository contains code accompanying our paper "FiD-ICL: A Fusion-in-Decoder Approach for Efficient In-Context Learning" (ACL 2023) [[Paper]](https://openreview.net/forum?id=7tO_uDjYyv).

FiD-ICL is inspired by [fusion-in-decoder models](https://github.com/facebookresearch/FiD/tree/main) designed for open-domain QA. In a meta-training setting, FiD-ICL outperforms the widely-adopted concatenation-based ICL, while being up to 10x faster at inference time. When compared to fine-tuning, the performance gap between FiD-ICL (gradient-free) and fine-tuning (gradient-based) is on average less than 3%.


### Quick Links
- [Configure Environment](#configure-environment)
- [Get Data](#get-data)
- [ICL Meta-Training](#icl-meta-training) 
- [ICL Meta-Testing](#icl-meta-testing)
- [Download Checkpoints](#download-checkpoints)
- [Contact Us](#contact-us)

### Configure Environment

We have included `requirements.txt` in this repository. Please run the following:
```bash
conda create -n fid-icl python=3.9
conda activate fid-icl
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Get Data

We use the following resources in our study:
* __Public Pool of Prompts (P3)__: We use the P3 data as meta-train and meta-test set.
* __BIG-bench__: We use 14 tasks from BIG-bench as meta-validation to select the best checkpoint during meta-training. We use these 14 tasks because they are also used in the original T0 paper.

Please check out the instructions to download the data in `t0/data_prep/README.md` and `big-bench/data_prep/README.md`.

### ICL Meta-Training

TODO :face_with_head_bandage:

### ICL Meta-Testing

TODO :face_with_head_bandage:

### Other Baselines

TODO :face_with_head_bandage:

We include example scripts for running the baselines in this repository.
* Zero-shot Evaluation
* Simple FT

### Download Checkpoints

Checkpoints are uploaded to huggingface hub. Their model identifier are listed in the table below. Will add Few-shot Learning

| Zero-shot | 
| :---      | 
|`google/t5-base-lm-adapt`|
|`qinyuany/my-t0-base`|
|`google/t5-large-lm-adapt`|
|`qinyuany/my-t0-large`|
|`google/t5-xl-lm-adapt`|
|`qinyuany/my-t0-3b`|


To run these models, please run the following:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
s
key = "qinyuany/my-t0-base" # or other model identifier

tokenizer = AutoTokenizer.from_pretrained(key)
model = AutoModelForSeq2SeqLM.from_pretrained(key)

```
### Contact Us

If you have any question, please submit an issue, or reach out to Qinyuan (qinyuany@usc.edu).

### Todos
* Update the bibkey once FiD-ICL is included in ACL Anthology.
* Clean the code and update meta-training, meta-testing section in README.
* Upload all checkpoints
* Upload poster and video, update links to this repo.