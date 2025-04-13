# Running scripts for yamyam-lab

`scripts/` contains various scripts used when building recommender system in `yamyam-lab` repository.


| Python script                          | Description                                                        |
|----------------------------------------|--------------------------------------------------------------------|
| `scripts/create_google_drive_token.py` | Used when creating token.json in ci                                |
| `scripts/download_candidate_result.py` | Used when downloading candidate generation or trained model result |

## How to download candidate generation or trained model result

Place credential file to authenticate google drive api to `credentials/` directory.

Refer to [this discussion](https://github.com/LearningnRunning/yamyam-lab/discussions/118#discussioncomment-12590729) about how to download credential json file from gcp.

Run following command depending on the embedding model and what you want to download, which is either of `candidates` or `models`.

- If you want to download candidate generation results, place `--download_file_type` argument as `candidates`.
- If you want to download trained model results with torch weight, logs, metric plot etc, place `--download_file_type` argument as `models`.

```bash
$ poetry run python3 scripts/download_result.py \
  --model_name "node2vec" \
  --download_file_type "candidates" \
  --latest \
  --credential_file_path_from_gcloud_console "PATH/TO/CREDENTIALS.json" \
  --reusable_token_path "PATH/TO/TOKEN.json"
```

Refer to description of each parameter.

| Parameter name                             | Description                                                                                |
|--------------------------------------------|--------------------------------------------------------------------------------------------|
| `model_name`                               | Name of embedding model (`node2vec` / `metapath2vec` / `graphsage` are allowed)            |
| `download_file_type`                       | File type you want to download (`candidates` / `models` are allowed)                       |
| `latest`                                   | Indicator whether downloading latest candidate generation results in selected model or not |
| `credential_file_path_from_gcloud_console` | Path to credential json file from gcp                                                      |
| `reusable_token_path`                      | Path to reusable token path                                                                |


After running script, check whether zip file is downloaded in `candidates/{model_name}` or `trained_models/{model_name}` successfully.

Latest version of each embedding model is given below. Note that identical version is applied both of candidate generation result and training result. (If zip file name is identical, they are generated from same training pipeline)

| Model name   | Latest version   |
|--------------|------------------|
| node2vec     | 202504070010.zip |
| metapath2vec | 202504011954.zip |
| graphsage    | 202504122051.zip |


## How to generate candidates from trained embedding model

Place credential file to authenticate google drive api to `credentials/` directory.

Refer to [this discussion](https://github.com/LearningnRunning/yamyam-lab/discussions/118#discussioncomment-12590729) about how to download credential json file from gcp.

You should specify path for trained pytorch weight and data object saved when embedding training.

Depending on the embedding model you want, different arguments are required.

Refer to description of each parameter.

| Parameter name        | Description                                                                     |
|-----------------------|---------------------------------------------------------------------------------|
| `model`               | Name of embedding model (`node2vec` / `metapath2vec` / `graphsage` are allowed) |
| `embedding_dim`       | Embedding dimension for trained model                                           |
| `data_obj_path`       | Path to data_object.pkl                                                         |
| `model_pt_path`       | Path to weight.pt                                                               |
| `weighted_edge`       | Indicator for weighted edge                                                     |
| `candidate_top_k`     | Number of candidates to generate                                                |
| `reusable_token_path` | Path to reusable token path                                                     |
| `use_metadata`        | Indicator for using metadata                                                    |
| `num_sage_layers`     | Number of sage layers                                                           |

### node2vec

```bash
$ poetry run python3 scripts/candidate/generate_candidate.py \
  --model node2vec \
  --embedding_dim 32 \
  --data_obj_path /PATH/TO/NODE2VEC/data_object.pkl \
  --model_pt_path /PATH/TO/NODE2VEC/weight.pt \
  --weighted_edge \
  --candidate_top_k 100 \
  --reusable_token_path PATH/TO/token.json
```

### metapath2vec
```bash
$ poetry run python3 scripts/candidate/generate_candidate.py \
  --model metapath2vec \
  --embedding_dim 32 \
  --data_obj_path /PATH/TO/METAPATH2VEC/data_object.pkl \
  --model_pt_path /PATH/TO/METAPATH2VEC/weight.pt \
  --weighted_edge \
  --use_metadata \
  --candidate_top_k 100 \
  --reusable_token_path PATH/TO/token.json
```


### graphsage

```bash
$ poetry run python3 scripts/candidate/generate_candidate.py \
  --model graphsage \
  --embedding_dim 32 \
  --data_obj_path /PATH/TO/GRAPHSAGE/data_object.pkl \
  --model_pt_path /PATH/TO/GRAPHSAGE/weight.pkl \
  --weighted_edge \
  --num_sage_layers 2 \
  --candidate_top_k 100 \
  --reusable_token_path PATH/TO/token.json
```
