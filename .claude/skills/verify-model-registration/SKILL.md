---
name: verify-model-registration
description: Verifies new models are properly registered in TrainerFactory, have Hydra configs, train.py routing, and implement all BaseTrainer abstract methods. Use after adding a new model.
disable-model-invocation: true
argument-hint: "[optional: specific model name to verify]"
---

# Model Registration Verification

## Purpose

Ensures new models are fully integrated into the training pipeline:

1. **Factory registration** — Model must be added to `TrainerFactory.TRAINER_REGISTRY` in `src/yamyam_lab/engine/factory.py`
2. **Hydra config** — Model must have a YAML config file under `config/models/{category}/`
3. **Train.py routing** — Model must be routed to the correct arg parser in `src/yamyam_lab/train.py`
4. **Trainer implementation** — Trainer class must implement all `BaseTrainer` abstract methods
5. **Import chain** — Factory must import the trainer class, and the trainer must import the model

## When to Run

- After adding a new model class in `src/yamyam_lab/model/`
- After creating a new trainer in `src/yamyam_lab/engine/`
- After modifying `TrainerFactory.TRAINER_REGISTRY`
- After modifying the arg parser routing in `train.py`
- Before creating a Pull Request that introduces a new model

## Related Files

| File | Purpose |
|------|---------|
| `src/yamyam_lab/engine/factory.py` | `TrainerFactory.TRAINER_REGISTRY` — model-to-trainer mapping |
| `src/yamyam_lab/engine/base_trainer.py` | Abstract methods: `load_data`, `build_model`, `build_metric_calculator`, `train_loop`, `evaluate_validation`, `evaluate_test` |
| `src/yamyam_lab/train.py` | Entry point with model-to-parser routing (`parse_args_graph`, `parse_args_als`, `parse_args`, `parse_args_multimodal_triplet`) |
| `src/yamyam_lab/tools/parse_args.py` | Argument parser definitions |
| `config/models/graph/` | Graph model configs (node2vec, graphsage, metapath2vec, lightgcn) |
| `config/models/mf/` | Matrix factorization model configs (als, svd_bias) |
| `config/models/embedding/` | Embedding model configs (multimodal_triplet) |
| `config/models/ranker/` | Ranker model configs (lightgbm, catboost) |
| `src/yamyam_lab/engine/als_trainer.py` | ALS trainer implementation |
| `src/yamyam_lab/engine/torch_trainer.py` | PyTorch MF trainer implementation |
| `src/yamyam_lab/engine/graph_trainer.py` | Graph model trainer implementation |
| `src/yamyam_lab/engine/base_embedding_trainer.py` | Base embedding trainer |
| `src/yamyam_lab/engine/multimodal_triplet_trainer.py` | Multimodal triplet trainer |

## Workflow

### Step 1: Extract Registered Models from Factory

**Tool:** Read

Read `src/yamyam_lab/engine/factory.py` and extract all keys from `TRAINER_REGISTRY`.

Current registered models:
- `als` -> `ALSTrainer`
- `node2vec` -> `GraphTrainer`
- `graphsage` -> `GraphTrainer`
- `metapath2vec` -> `GraphTrainer`
- `lightgcn` -> `GraphTrainer`
- `svd_bias` -> `TorchTrainer`
- `multimodal_triplet` -> `MultimodalTripletTrainer`

Check for any new models added to the registry.

**PASS:** Registry is consistent with imports at the top of the file.

**FAIL:** A model references a trainer class that is not imported, or an imported trainer is unused.

**Fix:** Add the missing import or remove the unused import.

### Step 2: Verify Hydra Config Exists for Each Model

**Tool:** Glob

For each model in `TRAINER_REGISTRY`, check that a config YAML exists:

```bash
ls config/models/graph/{node2vec,graphsage,metapath2vec,lightgcn}.yaml
ls config/models/mf/{als,svd_bias}.yaml
ls config/models/embedding/multimodal_triplet.yaml
```

For any newly registered model, check:

```bash
find config/models -name "{model_name}.yaml"
```

**PASS:** Every registered model has a corresponding YAML config file.

**FAIL:** A registered model has no YAML config.

**Fix:** Create `config/models/{category}/{model_name}.yaml` with required sections (preprocess, training, post_training).

### Step 3: Verify Train.py Routing

**Tool:** Read

Read `src/yamyam_lab/train.py` and verify that every model in `TRAINER_REGISTRY` is handled in the `if/elif` chain:

```python
# Current routing:
# graph models: ["node2vec", "graphsage", "metapath2vec", "lightgcn"] -> parse_args_graph()
# als: ["als"] -> parse_args_als()
# multimodal_triplet: ["multimodal_triplet"] -> parse_args_multimodal_triplet()
# torch models: ["svd_bias"] -> parse_args()
```

**PASS:** Every model in `TRAINER_REGISTRY` is routed to a parser in `train.py`.

**FAIL:** A registered model is not handled in `train.py` routing.

**Fix:** Add the model name to the appropriate parser branch, or create a new parser if needed.

### Step 4: Verify Trainer Implements All Abstract Methods

**Tool:** Read, Grep

Read `src/yamyam_lab/engine/base_trainer.py` to get the list of abstract methods:
- `load_data(self) -> None`
- `build_model(self) -> None`
- `build_metric_calculator(self) -> None`
- `train_loop(self) -> None`
- `evaluate_validation(self) -> None`
- `evaluate_test(self) -> None`

For each trainer class used by a new model, verify it implements all abstract methods:

```bash
grep -n "def load_data\|def build_model\|def build_metric_calculator\|def train_loop\|def evaluate_validation\|def evaluate_test" src/yamyam_lab/engine/{trainer_file}.py
```

**PASS:** All 6 abstract methods are implemented.

**FAIL:** One or more abstract methods are missing.

**Fix:** Implement the missing abstract methods in the trainer class.

### Step 5: Verify Import Chain Integrity

**Tool:** Grep

For each new model, verify the chain: `factory.py` imports trainer -> trainer imports model.

```bash
grep -n "from.*import.*{TrainerClass}" src/yamyam_lab/engine/factory.py
grep -n "from.*import\|import " src/yamyam_lab/engine/{trainer_file}.py | grep -i model
```

**PASS:** Import chain is complete from factory to trainer to model.

**FAIL:** Broken import chain.

**Fix:** Add missing imports.

## Output Format

```markdown
| # | Check | Status | Details |
|---|-------|--------|---------|
| 1 | Factory registration | PASS/FAIL | Models: [list], missing imports: [list] |
| 2 | Hydra config | PASS/FAIL | Missing configs: [list] |
| 3 | Train.py routing | PASS/FAIL | Unrouted models: [list] |
| 4 | Abstract methods | PASS/FAIL | Missing implementations: [list] |
| 5 | Import chain | PASS/FAIL | Broken chains: [list] |
```

## Exceptions

1. **Ranker models** — Models under `src/yamyam_lab/model/rank/` use a different entry point (`src/yamyam_lab/rerank.py` with Hydra) and do not need to be in `TrainerFactory.TRAINER_REGISTRY` or `train.py`
2. **Classic CF models** — Models under `src/yamyam_lab/model/classic_cf/` (user_based, item_based) may be used independently without factory registration
3. **Classification models** — Models under `src/yamyam_lab/model/classification/` may use separate entry points
4. **Shared trainers** — Multiple models can share the same trainer class (e.g., all graph models use `GraphTrainer`), so a new model may not need a new trainer
