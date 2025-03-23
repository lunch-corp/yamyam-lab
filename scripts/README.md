# Running scripts for yamyam-lab

`scripts/` contains various scripts used when building recommender system in `yamyam-lab` repository.

## Candidate scripts

### How to download candidate result

Place credential file to authenticate google drive api to `credentials/` directory.

Refer to [this discussion](https://github.com/LearningnRunning/yamyam-lab/discussions/118#discussioncomment-12590729) about how to download credential json file from gcp.

Run following command depending on the candidate generator model you want.

```bash
$ python3 scripts/candidate/download_candidate_result.py \
  --model_name "node2vec" \
  --latest \
  --credential_file_path_from_gcloud_console "PATH/TO/CREDENTIALS.json" \
  --reusable_token_path "PATH/TO/TOKEN.json"
```

Refer to description of each parameter.

| Parameter name                             | Description                                                                                |
|--------------------------------------------|--------------------------------------------------------------------------------------------|
| `model_name`                               | Name of candidate generator model (`node2vec` / `metapath2vec` / `graphsage` are allowed)  |
| `latest`                                   | Indicator whether downloading latest candidate generation results in selected model or not |
| `credential_file_path_from_gcloud_console` | Path to credential json file from gcp                                                      |
| `reusable_token_path`                      | Path to reusable token path                                                                |


After running script, check whether zip file is downloaded in `candidates/{model_name}` successfully.

Latest version of each candidate generator model is given below.

| Model name   | Latest version |
|--------------|----------------|
| node2vec     | TBD            |
| metapath2vec | TBD            |
| graphsage    | TBD            |
