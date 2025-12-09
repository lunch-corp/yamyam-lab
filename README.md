# yamyam-lab

This repository aims for developing recommender system using review data in kakao [map](https://map.kakao.com/).

## Setting up environment

We use [poetry](https://github.com/python-poetry/poetry) to manage dependencies of repository.

Use poetry with version `2.1.1`.

```shell
$ poetry --version
Poetry (version 2.1.1)
```

Python version should be `3.11.x`.

```shell
$ python --version
Python 3.11.11
```

If python version is lower than `3.11`, try installing required version using `pyenv`.

Create virtual environment.

```shell
$ poetry env activate
```

If your global python version is not 3.11, run following command.

```shell
$ poetry env use python3.11
```

You can check virtual environment path info and its executable python path using following command.

```shell
$ poetry env info
```

After setting up python version, just run following command which will install all the required packages from `poetry.lock`.

```shell
$ poetry install
```

## Setting up git hook

Set up automatic linting using the following commands:
```shell
# This command will ensure linting runs automatically every time you commit code.
pre-commit install
```

### Note

If you want to add package to `pyproject.toml`, please use following command.

```shell
$ poetry add "package==1.0.0"
```

Then, update `poetry.lock` to ensure that repository members share same environment setting.

```shell
$ poetry lock
```

---

## How to load review data using `google_drive.py`

To download `diner.csv`, `review.csv`, `reviewer.csv`, `diner_raw_category.csv`, follow below guideline.

1. File config:
   - Ensure that `DATA_FOLDER_ID` is defined in `.env` file. Currently, `DATA_FOLDER_ID` indicates google drive folder id where above 4 csv dataset are stored.
      ```dotenv
      DATA_FOLDER_ID=${DATA_FOLDER_ID}
      ```
   - The key values in the ``.env`` file will be removed from the README when shared publicly.

2. Download and Load Data:
   Use the following Python code to ensure the data files are available and load them into Pandas DataFrames:

   ```python
   from tools.google_drive import ensure_data_files
   import pandas as pd

   # Ensure required data files are available
   data_paths = ensure_data_files()

   # load data
   diner = pd.read_csv(data_paths["diner"])
   diner_category = pd.read_csv(data_paths["category"])


   diner = pd.merge(diner, diner_category, on="diner_idx", how="left")


   review = pd.read_csv(data_paths["review"])
   reviewer = pd.read_csv(data_paths["reviewer"])
   review_keyword = pd.read_csv(data_paths["review_keyword"])
   review = pd.merge(review, reviewer, on="reviewer_id", how="left")
   ```

3. Data Description:
   For detailed descriptions of the data (e.g., column names, data types, and content), refer to the [data/README.md file](data/README.md). This file provides comprehensive information about each dataset included in the project.

4. Requesting Data Access:
   If you are interested in running the code and need access to the data, please contact us at [leewook94@gmail.com](mailto:leewook94@gmail.com). We can provide the `DATA_FOLDER_ID` for the Google Drive folder containing the datasets.

---

## Implemented models

| Type                 | Algorithm       | 
|----------------------|-----------------|
| Baseline model       | Most Popular    |
| Baseline model       | ALS             |
| Baseline model       | SVD_Bias        |
| Candidate generation | node2vec        |
| Candidate generation | metapath2vec    |
| Candidate generation | graphsage       |
| Reranker             | lightgbm ranker |
| Reranker             | xgboost ranker  |

We are planning to generate candidate diners of each user using `candidate generation model` and rerank them using `reranker model`. Also, we will compare two-stage model results with baseline models.


---

## How to run training

Run from repo root (after `poetry install` and `.env` with `DATA_FOLDER_ID`):

```shell
# general form
poetry run python -m yamyam_lab.train --model <model_name> [options]
```

Quick examples:
- SVD_Bias: `poetry run python -m yamyam_lab.train --model svd_bias --epochs 10 --device cpu`

- ALS: `poetry run python -m yamyam_lab.train --model als --factors 100 --iterations 15`

- Node2Vec: `poetry run python -m yamyam_lab.train --model node2vec --epochs 10 --walk_length 20 --p 1 --q 1 --device cpu`

- GraphSAGE: `poetry run python -m yamyam_lab.train --model graphsage --num_sage_layers 2 --num_neighbor_samples 3`

- LightGCN: `poetry run python -m yamyam_lab.train --model lightgcn --num_lightgcn_layers 3 --drop_ratio 0.1`

- LightGBM Ranker: `poetry run python -m yamyam_lab.rerank models/ranker=lightgbm`

Note: `--model` is required. Supported values: `svd_bias`, `als`, `node2vec`, `metapath2vec`, `graphsage`, `lightgcn`. See `src/yamyam_lab/tools/parse_args.py` for all available flags (e.g., `--batch_size`, `--lr`, `--save_candidate`, `--config_root_path`, etc.).

---

## Experiment results

We evaluate model results in two aspects.

* First of all, we measure performance of candidate generation model using `recall` metric.
  * For candidate generation model, it is important to achieve high hit ratio, i.e., recall.
  * After achieving high recall, detail ranking will be done via reranker model.
* Next, we measure performance of ranking using `map` and `ndcg` metric.
  * With `map` and `ndcg`, we evaluate ranking ability of models whether liked items by users are ranked with higher rank or not.
* Note that for comparison, candidate generation models are also evaluated with ranking metric.

For detail description of each metric, please refer to [discussion](https://github.com/LearningnRunning/yamyam-lab/discussions/74).

For detail experiment results, please refer to [discussion](https://github.com/lunch-corp/yamyam-lab/discussions/173).

---

## Project code lint

We use ruff lint for project code consistency. Run following command if ruff lint check passes.

```shell
$ make lint
```

You should update code corresponding to ruff's guide, otherwise ci test won't pass.

---

## How to run pytest

After building environment setting correctly, just run the following command.

```shell
$ make test
```
