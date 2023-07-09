### Install BIG-bench

```bash
cd
git clone https://github.com/google/BIG-bench.git
cd BIG-bench
python setup.py sdist
pip install -e .
```

### Get the task files

```bash
# come back to this directory
python get_data.py
python process_strategyqa.py
```