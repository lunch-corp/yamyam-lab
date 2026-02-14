# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

yamyam-lab is a recommender system for restaurants using review data from Kakao Map. It implements a two-stage recommendation pipeline: candidate generation followed by reranking.

## Common Commands

```bash
# Install dependencies
poetry install

# Run linting (format + check with auto-fix prompt)
make lint

# Run tests
make test

# Run a single test file
poetry run pytest tests/path/to/test_file.py

# Run a single test
poetry run pytest tests/path/to/test_file.py::test_function_name -v

# Train models
poetry run python -m yamyam_lab.train --model <model_name> [options]
# Examples:
#   poetry run python -m yamyam_lab.train --model svd_bias --epochs 10 --device cpu
#   poetry run python -m yamyam_lab.train --model node2vec --epochs 10 --walk_length 20 --p 1 --q 1
#   poetry run python -m yamyam_lab.train --model graphsage --num_sage_layers 2 --num_neighbor_samples 3

# Train reranker (uses Hydra config)
poetry run python -m yamyam_lab.rerank models/ranker=lightgbm

# Run Streamlit dashboard
poetry run streamlit run apps/main.py
```

## Architecture

### Training Pipeline

The training system uses Factory and Template Method patterns:

- `src/yamyam_lab/train.py` - Entry point that routes to appropriate trainer via `TrainerFactory`
- `src/yamyam_lab/engine/factory.py` - Maps model names to trainer classes
- `src/yamyam_lab/engine/base_trainer.py` - Abstract base with template method defining workflow: load_configs → setup_logger → load_data → build_model → build_metric_calculator → train_loop → evaluate_validation → evaluate_test → post_process

Trainer implementations:
- `TorchTrainer` - For `svd_bias`
- `GraphTrainer` - For `node2vec`, `metapath2vec`, `graphsage`, `lightgcn`
- `ALSTrainer` - For `als`

### Reranking Pipeline

- `src/yamyam_lab/rerank.py` - Hydra-based entry point for training LightGBM/XGBoost rankers
- Uses `RankerDatasetLoader` for data preparation and `RankerMetricCalculator` for evaluation
- Configuration via `config/train.yaml` and `config/models/ranker/`

### Key Modules

- `src/yamyam_lab/model/` - Model implementations (graph/, mf/, rank/, classic_cf/)
- `src/yamyam_lab/data/` - Data loading and preprocessing
- `src/yamyam_lab/evaluation/` - Metrics (recall for candidates, MAP/NDCG for ranking)
- `src/yamyam_lab/features/` - Feature engineering
- `src/yamyam_lab/tools/` - Utilities including Google Drive data loading, config parsing, logging

### Configuration

- Hydra configs in `config/` directory
- Model configs: `config/models/{graph,mf,ranker}/`
- Data config: `config/data/dataset.yaml`
- Preprocess config: `config/preprocess/`

### Data Access

Data is stored in Google Drive. Set `DATA_FOLDER_ID` in `.env` file, then use:
```python
from tools.google_drive import ensure_data_files
data_paths = ensure_data_files()
```

## Code Style

- Python 3.11
- Ruff for linting/formatting (line length 88, Black-compatible)
- Pre-commit hooks: run `pre-commit install` after cloning

## Skills

| Skill | Description |
|-------|-------------|
| manage-skills | Analyzes session changes to detect missing verification skills. Dynamically explores existing skills, creates new skills or updates existing ones, and manages CLAUDE.md. |
| verify-implementation | Sequentially executes all verify skills in the project to generate an integrated verification report. Use after feature implementation, before PRs, or during code review. |
| verify-test-coverage | Verifies new models/trainers/pipelines have corresponding unit and integration tests. |
| verify-code-convention | Validates PEP 8 naming, type hints, ruff compliance, and import ordering. |
| verify-model-registration | Ensures new models are registered in factory, have configs, train.py routing, and implement abstract methods. |
| verify-config-consistency | Validates Hydra config files have required sections, correct YAML structure, and resolvable references. |
