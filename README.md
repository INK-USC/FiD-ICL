## FiD-ICL: A Fusion-in-Decoder Approach for Efficient In-Context Learning


This repository contains code accompanying our paper "FiD-ICL: A Fusion-in-Decoder Approach for Efficient In-Context Learning" (ACL 2023) [[Paper]](https://openreview.net/forum?id=7tO_uDjYyv).

FiD-ICL is inspired by [fusion-in-decoder models](https://github.com/facebookresearch/FiD/tree/main) designed for open-domain QA. In a meta-training setting, FiD-ICL outperforms the widely-adopted concatenation-based ICL, while being up to 10x faster at inference time. When compared to fine-tuning, the performance gap between FiD-ICL (gradient-free) and fine-tuning (gradient-based) is on average less than 3%.


### Quick Links
- [Configure Environment](#configure-environment)
- [Get Data](#get-data)
- [Meta-Training](#meta-training) 
- [Meta-Testing](#meta-testing)
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

Please check out the instructions in `t0/data_prep/README.md` and `t0/data_prep/README.md`.

#### Public Pool of Prompts (P3) data
We use the P3 data as meta-train and meta-test set

#### BIG-bench data
We use 14 tasks from BIG-bench as meta-validation to select the best checkpoint during meta-training. We use these 14 tasks because they are also used in the original T0 paper.

### Meta-Training

### Meta-Testing

### Download Checkpoints

### Contact Us

If you have any question, please submit an issue, or reach out to Qinyuan (qinyuany@usc.edu).

If you used our code in your study, or find our paper useful, please cite us with the following BibTeX.

<details>
<summary>BibTeX</summary>

```
@unpublished{              
    ye2023fidicl:,              
    title={FiD-ICL: A Fusion-in-Decoder Approach for Efficient In-Context Learning},              
    author={Qinyuan Ye,Iz Beltagy,Matthew E. Peters,Xiang Ren,Hannaneh Hajishirzi},              
    journal={OpenReview Preprint},              
    year={2023}, 
}
```
</details>

<br>

### Todos
* Update the bibkey once FiD-ICL is included in ACL Anthology.
