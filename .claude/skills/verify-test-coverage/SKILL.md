---
name: verify-test-coverage
description: Verifies that new models, trainers, and pipeline components have corresponding unit tests and integration tests. Use after adding new models or modifying training pipeline.
disable-model-invocation: true
argument-hint: "[optional: specific model or module name]"
---

# Test Coverage Verification

## Purpose

Ensures all production code in the ML pipeline has adequate test coverage:

1. **Model test coverage** — Every model registered in `TrainerFactory.TRAINER_REGISTRY` must have an integration test in `tests/model/test_train.py`
2. **Module test coverage** — New modules under `src/yamyam_lab/` must have corresponding test files under `tests/`
3. **Test naming convention** — Test files must follow `test_{module_name}.py`, test classes `Test{ComponentName}`, test methods `test_{behavior}`
4. **Fixture completeness** — New models requiring custom config must have fixtures in `tests/conftest.py`

## When to Run

- After adding a new model to `TrainerFactory.TRAINER_REGISTRY`
- After creating a new trainer class in `src/yamyam_lab/engine/`
- After adding new modules under `src/yamyam_lab/` (data, evaluation, features, model, etc.)
- After modifying `BaseTrainer` abstract methods
- Before creating a Pull Request

## Related Files

| File | Purpose |
|------|---------|
| `src/yamyam_lab/engine/factory.py` | `TRAINER_REGISTRY` — source of truth for registered models |
| `src/yamyam_lab/engine/base_trainer.py` | Abstract base trainer defining required methods |
| `tests/model/test_train.py` | Integration tests for all models via `TrainerFactory` |
| `tests/conftest.py` | Shared fixtures (`setup_config`, `setup_als_config`, `mock_load_dataset`, etc.) |
| `tests/model/test_multimodal_triplet.py` | Model-specific unit tests example |
| `tests/model/test_reranker.py` | Reranker test example |
| `tests/model/test_category_classifier.py` | Classifier test example |
| `tests/data/test_dataset.py` | Data loader tests |
| `tests/evaluation/test_metric.py` | Metric calculation tests |
| `tests/features/test_diner_menu_functions.py` | Feature engineering tests |
| `tests/loss/test_infonce.py` | Loss function tests |
| `tests/preprocess/test_category_preprocess.py` | Preprocessing tests |

## Workflow

### Step 1: Check All Registered Models Have Integration Tests

**Tool:** Grep, Read

Extract all model names from `TRAINER_REGISTRY`:

```bash
grep -oP '"(\w+)":\s*\w+' src/yamyam_lab/engine/factory.py
```

Then verify each model name appears in `tests/model/test_train.py` as a parametrized test case or dedicated test class:

```bash
grep -n 'setup_config\|setup_als_config\|"als"\|"node2vec"\|"graphsage"\|"metapath2vec"\|"lightgcn"\|"svd_bias"\|"multimodal_triplet"' tests/model/test_train.py
```

**PASS:** Every model in `TRAINER_REGISTRY` has a corresponding test (either parametrized in `test_train.py` or in a dedicated test file like `test_multimodal_triplet.py`).

**FAIL:** A model is registered in the factory but has no integration test.

**Fix:** Add a parametrized test case to the appropriate test class in `tests/model/test_train.py`, or create a dedicated test file `tests/model/test_{model_name}.py`.

### Step 2: Check New Source Modules Have Test Files

**Tool:** Glob, Grep

List all Python modules under `src/yamyam_lab/` (excluding `__init__.py`, `__pycache__`):

```bash
find src/yamyam_lab -name "*.py" -not -name "__init__.py" -not -path "*__pycache__*" | sort
```

List all test files:

```bash
find tests -name "test_*.py" | sort
```

For each source module directory (`data/`, `evaluation/`, `features/`, `model/`, `engine/`, etc.), verify at least one corresponding test file exists under `tests/`.

**PASS:** Every source module directory with production logic has at least one test file.

**FAIL:** A source module directory has no corresponding test file under `tests/`.

**Fix:** Create `tests/{module}/test_{component}.py` with appropriate tests.

### Step 3: Check Test File Naming Convention

**Tool:** Glob

```bash
find tests -name "*.py" -not -name "test_*" -not -name "conftest.py" -not -name "__init__.py" -not -name "utils.py" -not -path "*__pycache__*"
```

**PASS:** All test files follow `test_{name}.py` naming (except `conftest.py`, `utils.py`, `__init__.py`).

**FAIL:** Test files found that don't follow the naming convention.

**Fix:** Rename to `test_{descriptive_name}.py`.

### Step 4: Check Test Class and Method Naming

**Tool:** Grep

```bash
grep -rn "^class " tests/ --include="*.py" | grep -v "^.*:class Test"
```

```bash
grep -rn "def test" tests/ --include="*.py" | grep -v "def test_"
```

**PASS:** All test classes use `Test{Name}` and all test methods use `test_{description}`.

**FAIL:** Classes or methods not following naming convention.

**Fix:** Rename to `Test{ComponentName}` / `test_{behavior_description}`.

### Step 5: Check New Models Have Config Fixtures

**Tool:** Read

Read `tests/conftest.py` and check that each model type has a fixture that provides its config. Models using `setup_config` must be listed in its parametrize options. Models with unique configs (like `als`, `multimodal_triplet`) must have their own fixture.

**PASS:** Every registered model has a config fixture available.

**FAIL:** A model is registered but has no config fixture.

**Fix:** Add a fixture to `tests/conftest.py` following existing patterns (`setup_config`, `setup_als_config`, `multimodal_triplet_config`).

## Output Format

```markdown
| # | Check | Status | Details |
|---|-------|--------|---------|
| 1 | Registered model integration tests | PASS/FAIL | List of untested models |
| 2 | Module test coverage | PASS/FAIL | List of uncovered modules |
| 3 | Test file naming | PASS/FAIL | List of misnamed files |
| 4 | Test class/method naming | PASS/FAIL | List of violations |
| 5 | Config fixtures | PASS/FAIL | List of missing fixtures |
```

## Exceptions

1. **Utility modules** — `src/yamyam_lab/tools/` helper files (e.g., `config.py`, `logger.py`, `parse_args.py`) do not require dedicated test files unless they contain complex logic
2. **`__init__.py` files** — These are re-export files and do not need tests
3. **Notebook and script files** — Files in `notebook/`, `apps/`, or one-off scripts are exempt
4. **Abstract base classes** — `base_trainer.py`, `base_embedding.py` etc. are tested indirectly through concrete implementations
5. **Models tested in dedicated files** — A model does not need to appear in `test_train.py` if it has its own comprehensive test file (e.g., `test_multimodal_triplet.py`)
