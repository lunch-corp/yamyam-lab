# Running scripts for yamyam-lab

## Candidate scripts

### How to download candidate result

Place credential file to authenticate google drive api to `credentials/` directory.

Run following command depending on the candidate generator model you want.

```bash
$ python3 scripts/candidate/download_candidate_result.py \
  --model_name "node2vec" \
  --latest \
  --credential_file_path_from_gcloud_console "PATH/TO/CREDENTIALS.json" \
  --reusable_token_path "PATH/TO/TOKEN.json"
```

Check whether zip file is downloaded in `candidates/{model_name}`.