## FiD-ICL: A Fusion-in-Decoder Approach for Efficient In-Context Learning


This repository contains code accompanying our paper "FiD-ICL: A Fusion-in-Decoder Approach for Efficient In-Context Learning" (ACL 2023) [[Paper]](https://openreview.net/forum?id=7tO_uDjYyv).

FiD-ICL is inspired by [fusion-in-decoder models](https://github.com/facebookresearch/FiD/tree/main) designed for open-domain QA. In a meta-training setting, FiD-ICL outperforms the widely-adopted concatenation-based ICL, while being up to 10x faster at inference time. When compared to fine-tuning, the performance gap between FiD-ICL (gradient-free) and fine-tuning (gradient-based) is on average less than 3%.


### Quick Links
- [Environment](#environment)
- [Data](#data)
- [ICL Meta-Training](#icl-meta-training) 
- [ICL Meta-Testing](#icl-meta-testing)
- [Other Baselines](#other-baselines)
- [Download Checkpoints](#download-checkpoints)
- [Contact Us](#contact-us)

### Environment

We have included `requirements.txt` in this repository. Please run the following:
```bash
conda create -n fid-icl python=3.9
conda activate fid-icl
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Data

We use the following resources in our study:
* __Public Pool of Prompts (P3)__: We use the P3 data as meta-train and meta-test set.
* __BIG-bench__: We use 14 tasks from BIG-bench as meta-validation to select the best checkpoint during meta-training. We use these 14 tasks because they are also used in the original T0 paper.

Please check out the instructions to download the data in `t0/data_prep/README.md` and `big-bench/data_prep/README.md`.

### ICL Meta-Training

If you only want to download the resulting model and reproduce the evaluation part, you don't need to run meta-training because model checkpoints are ported to huggingface hub. Go to [ICL Meta-Testing](#icl-meta-testing) below.

The following script will train a FiD-ICL model initializing the model with T5-LM-XL (This is the best performing ICL model in the paper).
```bash
cd encdec
python run_fid.py -c runs/metatrain/fid_t5_xl.json
```

Use `run_icl.py` and `run_ensemble.py` to meta-train a model with Concat-ICL and Ensemble-ICL. There are sample configurations in the `runs/metatrain` directory.

### ICL Meta-Testing

The following script will evaluate the FiD-ICL trained from T5-LM-XL.
```bash
python run_fid_eval.py -c runs/metatest/fid_t5_xl.json 
```

Similarly, use `run_icl_eval.py` and `run_ensemble_eval.py` to meta-test a model trained with Concat-ICL and Ensemble-ICL. There are sample configurations in the `runs/metatest` directory.

### Other Baselines

We include example scripts for running the baselines in this repository.

#### Zero-shot Evaluation (Re-trained T0 models)
```
python run_t0eval.py -c runs/metatest/t0_3b.json
# results will be in eval_logs/t0_eval/my-t0-3b/results_t0_template.csv
```
#### Simple FT
```
python run_t0fewshot.py -c runs/metatest/ft_t0_3b.json
```
#### T-Few FT
The only difference in experiment setting between us and the original T-Few paper is that we control the number of shots to be 16.
Download the T-Few repository (https://github.com/r-three/t-few); Copy and paste the data files in `t0/data_fewshot` to T-Few directory and change the data path in T-Few code.

### Download Checkpoints

Checkpoints are uploaded to huggingface hub. Their model identifier are listed in a table in `CHECKPOINTS.md` ([link](CHECKPOINTS.md)).

You can find these models on huggingface hub. If the identifier is `qinyuany/my-t0-3b` then the webpage is [https://huggingface.co/qinyuany/my-t0-3b](https://huggingface.co/qinyuany/my-t0-3b)

To use these models, please run the following:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
identifier = "qinyuany/my-t0-base" # or other model identifier

tokenizer = AutoTokenizer.from_pretrained(identifier)
model = AutoModelForSeq2SeqLM.from_pretrained(identifier)

```
### Contact Us

If you have any question, please submit an issue, or reach out to Qinyuan (qinyuany@usc.edu).
