# yamyam-lab

This repository aims for developing recommender system using review data in kakao [map](https://map.kakao.com/).

## Environment setting

We use [poetry](https://github.com/python-poetry/poetry) to manage dependencies of repository.

It is recommended that latest version of poetry should be installed in advance.

```shell
$ poetry --version
Poetry (version 1.8.5)
```

Python version should be higher than `3.11`.

```shell
$ python --version
Python 3.11.11
```

If python version is lower than `3.11`, try installing required version using `pyenv`.

Create virtual environment.

```shell
$ poetry shell
```

After setting up python version, just run following command which will install all the required packages from `poetry.lock`.

```shell
$ poetry install
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

아래는 `README.md`에 추가할 설명글입니다. `google_drive.py`를 활용하여 데이터를 `diner`, `review` 데이터프레임으로 로드하는 방법을 명확히 안내합니다:

---

## Load Data using `google_drive.py`

1. File config:
   - Environment Variables (.env File)
      Create a .env file in the root directory of the project. This file stores Google Drive file IDs and their respective local file paths. Add the following content to the .env file:
      ```dotenv
      DINER_FILE_ID=1-sMiojVncUyA7qCwuI3U_Lmkx0SjN3T6
      REVIEW_FILE_ID=1OVmMUM5b_He6QDyaD8iGMJbLK_QGHnZs
      REVIEWER_FILE_ID=1b986HsOhgCSjJUif5DDwXaOoYTbx-TdE
      CATEGORY_FILE_ID=1gnURUQCgN4Nmw5_F82z2r2pWHNX8XlPd
      ```

   - YAML Configuration File
      Ensure the config/data/google_drive.yaml file is properly set up. Example content:
      ```yaml
      local_paths:
         diner: "data/diner.csv"
         review: "data/review.csv"
         reviewer: "data/reviewer.csv"
         category: "data/diner_raw_category.csv"
      ```
   - The key values in the ``.env`` file will be removed from the README when shared publicly.

2. Download and Load Data:
   Use the following Python code to ensure the data files are available and load them into Pandas DataFrames:

   ```python
   from tools.google_drive import ensure_data_files
   import pandas as pd

   # Ensure required data files are available
   data_paths = ensure_data_files()

   # Load data into Pandas DataFrames
   diner = pd.read_csv(data_paths["diner"])
   review = pd.read_csv(data_paths["review"])
   
   # Merge review and reviewer data
   review = pd.merge(review, pd.read_csv(data_paths["reviewer"]), on="reviewer_id", how="left")
   ```

3. Data Description:
   For detailed descriptions of the data (e.g., column names, data types, and content), refer to the [data/README.md file](data/README.md). This file provides comprehensive information about each dataset included in the project.


## Experiment results

### CASE 1) Without candidates

Below are metric results without any candidate filtering.

* Recommendations are generated at user's level.
* This inference does not consider user's current location.

| Algorithm | Task         | mAP@3   | mAP@7   | mAP@10  | mAP@20  | NDCG@3  | NDCG@7  | NDCG@10 | NDCG@20 |
| --------- | ------------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| SVD       | Regression   | TBD     | TBD     | TBD     | TBD     | TBD     | TBD     | TBD     | TBD     |
| node2vec  | Unsupervised | 0.00803 | 0.00736 | 0.00743 | 0.00761 | 0.01253 | 0.01709 | 0.02082 | 0.02972 |

### CASE 2) With candidates filtering with near 1km diners.

Below are metric results with candidate filtering.

* Recommendations are generated at each data level in validation dataset.
* This inference regards diner's location as user's current location which actually cannot be obtained.

| Algorithm | Task         | ranked_prec@3 | ranked_prec@7 | ranked_prec@10 | ranked_prec@20 |
| --------- | ------------ | ------------- | ------------- | -------------- | -------------- |
| SVD       | Regression   | TBD           | TBD           | TBD            | TBD            |
| node2vec  | Unsupervised | 0.10109       | 0.14015       | 0.16174        | 0.21334        |


### CASE 3) Candidates generation models

| Algorithm | Task         | recall@100 | recall@300 | recall@500 |
| --------- | ------------ | ---------- | ---------- | ---------- |
| node2vec  | Unsupervised | 0.50612    | 0.74211    | 0.8439     |


### CASE 4) With two-step recommendations

| Candidate model | number of candidates | Reranking model | Task | ranked_prec@3 | ranked_prec@7 | ranked_prec@10 | ranked_prec@20 |
| --------------- | -------------------- | --------------- | ---- | ------------- | ------------- | -------------- | -------------- |
| node2vec        | TBD                  | lightgbm ranker | TBD  | TBD           | TBD           | TBD            | TBD            |

## Commit Guide
- feat: Add a new feature
- fix: Fix a bug
- docs: Update documentation
- style: Change code style (e.g., formatting, missing semicolons)
- design: Modify user interface design (e.g., CSS changes)
- test: Add or refactor test code
- refactor: Refactor production code
- build: Modify build files
- ci: Update CI configuration files
- perf: Improve performance
- chore: Minor updates or build maintenance
- rename: Rename files or folders only
- remove: Delete files only


## Project code lint

We use ruff lint for project code consistency. Run following command if ruff lint check passes.

```shell
$ make lint
```

You should update code corresponding to ruff's guide, otherwise ci test won't pass.

## How to run pytest

After building environment setting correctly, just run the following command.

```shell
$ make test
```