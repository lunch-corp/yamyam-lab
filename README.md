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

## Experiment results

### CASE 1) Without candidates

Below are metric results without any candidate filtering.

* Recommendations are generated at user's level.
* This inference does not consider user's current location.

|Algorithm|Task|mAP@3|mAP@7|mAP@10|mAP@20|NDCG@3|NDCG@7|NDCG@10|NDCG@20|
|----------------|---------|------|------|------|-------|-------|-------|-------|-------|
|SVD|Regression|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|node2vec|Unsupervised|0.00803|0.00736|0.00743|0.00761|0.01253|0.01709|0.02082|0.02972

### CASE 2) With candidates filtering with near 1km diners.

Below are metric results with candidate filtering.

* Recommendations are generated at each data level in validation dataset.
* This inference regards diner's location as user's current location which actually cannot be obtained.

|Algorithm|Task|ranked_prec@3|ranked_prec@7|ranked_prec@10|ranked_prec@20|
|----------------|---------|------|------|------|-------|
|SVD|Regression|TBD|TBD|TBD|TBD|
|node2vec|Unsupervised|0.10109|0.14015|0.16174|0.21334|


### CASE 3) Candidates generation models

|Algorithm|Task|recall@100|recall@300|recall@500|
|----------------|---------|------|------|------|
|node2vec|Unsupervised|0.50612|0.74211|0.8439|


### CASE 4) With two-step recommendations

|Candidate model|number of candidates|Reranking model|Task|ranked_prec@3|ranked_prec@7|ranked_prec@10|ranked_prec@20|
|---------------|--------------------|---------------|----|-------------|-------------|--------------|--------------|
|node2vec|TBD|lightgbm ranker|TBD|TBD|TBD|TBD|TBD|

## Commit 가이드
- feat: 새로운 기능 추가
- fix: 버그 수정
- docs: 문서 수정
- style: 코드 스타일 변경 (코드 포매팅, 세미콜론 누락 등)
- design: 사용자 UI 디자인 변경 (CSS 등)
- test: 테스트 코드, 리팩토링 (Test Code)
- refactor: 리팩토링 (Production Code)
- build: 빌드 파일 수정
- ci: CI 설정 파일 수정
- perf: 성능 개선
- chore: 자잘한 수정이나 빌드 업데이트
- rename: 파일 혹은 폴더명을 수정만 한 경우
- remove: 파일을 삭제만 한 경우
