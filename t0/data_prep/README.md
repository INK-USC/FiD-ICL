### Download P3 data from huggingface datasets

```bash
python download_data.py 0 660
python download_eval_data.py
```

Because there are 660 tasks registered in huggingface datasets `bigscience/P3`, "0 660" means the start and end index.

To speed up things, a cranky (:joy:) multi-processing version would be launch the script multiple times with 0-100, 101-200, 201-300, etc. By doing this, data downloading can be finished overnight. If something fails, restart from the failing index (e.g., run 50-100 if it fails at 50). 

### Subsample the data
The complete P3 dataset is huge. To make our experiments tractable, we subsample the data to have 50k exmaples at most for each task.

```bash
python subsample.py data_small 50000
```

Sometimes I just want to test my code quickly. But loading al data from `data_small` can be slow. So I prepared a tiny version for this purpose.

```bash
python subsample.py data_tiny 1000
```

### Getting StoryCloze Data
1. Get the [dataset](https://cs.rochester.edu/nlp/rocstories/) by filling out [this form](https://docs.google.com/forms/d/e/1FAIpQLSe83zPs21IGH9-HC1SuUa2hfyopJOHgTHft--Ne4SOj0VoViA/viewform?c=0&w=1). In the email you receive, download Story Cloze Test Spring 2016 set (val set + test set) as csv files.

2. You should have two files: `cloze_test_test__spring2016 - cloze_test_ALL_test.csv` and `cloze_test_val__spring2016 - cloze_test_ALL_val.csv`. Place then in the `storycloze` subdirectory in this directory.

3. run `python story_cloze.py` which will save the few-shot data to `data_fewshot` and the evaluation data to `data_eval`.
