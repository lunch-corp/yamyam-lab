---
name: verify-config-consistency
description: Validates Hydra config files have required sections, correct YAML structure, and that referenced _target_ classes and data paths exist. Use after adding or modifying model configs.
disable-model-invocation: true
argument-hint: "[optional: specific config file or model name]"
---

# Config Consistency Verification

## Purpose

Ensures Hydra configuration files are valid and consistent with the codebase:

1. **Required sections** — Model configs must have the necessary sections (`preprocess`, `training`, `post_training`)
2. **`_target_` class resolution** — Classes referenced by `_target_` must exist in the codebase
3. **Config-to-model alignment** — Config file names must match registered model names
4. **Evaluation config** — Training configs must include `evaluation` sub-section with `top_k_values_for_pred` and `recommend_batch_size`
5. **Post-training file names** — `post_training.file_name` must include required output file keys

## When to Run

- After creating a new model config YAML
- After modifying existing config files
- After renaming or moving model classes
- After changing evaluation metrics or output file patterns
- Before creating a Pull Request

## Related Files

| File | Purpose |
|------|---------|
| `config/train.yaml` | Main Hydra config with defaults |
| `config/data/dataset.yaml` | Dataset configuration |
| `config/preprocess/preprocess.yaml` | Preprocessing configuration |
| `config/models/graph/node2vec.yaml` | Graph model config example |
| `config/models/graph/graphsage.yaml` | Graph model config example |
| `config/models/graph/metapath2vec.yaml` | Graph model config example |
| `config/models/graph/lightgcn.yaml` | Graph model config example |
| `config/models/mf/als.yaml` | MF model config example |
| `config/models/mf/svd_bias.yaml` | MF model config example |
| `config/models/embedding/multimodal_triplet.yaml` | Embedding model config example |
| `config/models/ranker/lightgbm.yaml` | Ranker config with `_target_` |
| `config/models/ranker/catboost.yaml` | Ranker config with `_target_` |
| `src/yamyam_lab/engine/factory.py` | Source of truth for model names |
| `src/yamyam_lab/tools/config.py` | Config loading logic |

## Workflow

### Step 1: Verify Config Files Exist for All Registered Models

**Tool:** Glob, Read

Read `src/yamyam_lab/engine/factory.py` to get all model names from `TRAINER_REGISTRY`. Then check each model has a config:

```bash
find config/models -name "*.yaml" | sort
```

Cross-reference with registered models. Each model in `TRAINER_REGISTRY` must have a matching `config/models/{category}/{model_name}.yaml`.

**PASS:** Every registered model has a config file.

**FAIL:** A registered model is missing its config file.

**Fix:** Create `config/models/{category}/{model_name}.yaml` following the pattern of existing configs in the same category.

### Step 2: Verify Required Config Sections

**Tool:** Read

For each model config (non-ranker), verify these sections exist:

**Standard models (graph, mf, embedding):**
- `preprocess.data` — Must include time point fields: `train_time_point`, `val_time_point`, `test_time_point`, `end_time_point`
- `training` — Must include `lr` or equivalent hyperparameters
- `training.evaluation` — Must include `recommend_batch_size`, `top_k_values_for_pred`
- `post_training.file_name` — Must include at minimum: `log`, `weight` (or equivalent)
- `post_training.candidate_generation` — Must include `top_k`

**Ranker models:**
- `_target_` — Must reference a valid class path
- `features` — List of feature names
- `params` — Model-specific hyperparameters

**PASS:** All required sections present.

**FAIL:** Missing required sections.

**Fix:** Add the missing sections following existing config patterns.

### Step 3: Verify `_target_` Class References

**Tool:** Grep

For ranker configs that use Hydra's `_target_` for class instantiation:

```bash
grep -rn "_target_" config/models/ --include="*.yaml"
```

For each `_target_` value, verify the class exists:

```bash
grep -rn "class {ClassName}" src/ --include="*.py"
```

**PASS:** All `_target_` classes can be resolved to actual Python classes.

**FAIL:** A `_target_` references a non-existent class.

**Fix:** Update the `_target_` path or create the missing class.

### Step 4: Verify Config-Model Name Alignment

**Tool:** Glob

```bash
find config/models -name "*.yaml" | sort
```

Each YAML filename (without extension) should match either:
- A model name in `TRAINER_REGISTRY`, OR
- A ranker model name used via Hydra override (e.g., `lightgbm`, `catboost`)

```bash
grep -oP '"(\w+)"' src/yamyam_lab/engine/factory.py
```

**PASS:** All config filenames align with model names.

**FAIL:** Config file exists with no matching model, or vice versa.

**Fix:** Rename the config file to match the model name, or register the model if it's new.

### Step 5: Verify YAML Syntax

**Tool:** Bash

```bash
poetry run python -c "
import yaml, glob
for f in glob.glob('config/**/*.yaml', recursive=True):
    try:
        with open(f) as fp:
            yaml.safe_load(fp)
    except yaml.YAMLError as e:
        print(f'INVALID: {f}: {e}')
"
```

**PASS:** All YAML files parse without errors.

**FAIL:** YAML syntax errors found.

**Fix:** Fix the YAML syntax errors.

### Step 6: Verify Evaluation Config Consistency

**Tool:** Read

Read all model configs and verify `training.evaluation` sections use consistent key names:

- `recommend_batch_size` (not `batch_size` or `eval_batch_size`)
- `top_k_values_for_pred` (not `top_k` or `k_values`)
- `top_k_values_for_candidate` (optional, for candidate generation models)

**PASS:** All configs use consistent evaluation key names.

**FAIL:** Inconsistent key names found.

**Fix:** Standardize to the project's naming convention.

## Output Format

```markdown
| # | Check | Status | Details |
|---|-------|--------|---------|
| 1 | Config file existence | PASS/FAIL | Missing configs: [list] |
| 2 | Required sections | PASS/FAIL | Missing sections: [list] |
| 3 | _target_ resolution | PASS/FAIL | Unresolvable targets: [list] |
| 4 | Name alignment | PASS/FAIL | Mismatches: [list] |
| 5 | YAML syntax | PASS/FAIL | Invalid files: [list] |
| 6 | Evaluation consistency | PASS/FAIL | Inconsistencies: [list] |
```

## Exceptions

1. **Main config** — `config/train.yaml` is the Hydra root config and follows a different structure (defaults list); it does not need model-specific sections
2. **Data and preprocess configs** — `config/data/dataset.yaml` and `config/preprocess/preprocess.yaml` have their own structure and are not model configs
3. **Log config** — `config/log/logging.yaml` is for logging configuration
4. **Optional sections** — `model` section (model architecture params) is only required for neural models, not for `als` or ranker models
5. **Embedding models** — May have a `data` section with paths to preprocessed features instead of the standard `preprocess.data` section
