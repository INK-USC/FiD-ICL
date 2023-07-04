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