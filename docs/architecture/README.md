# Two-Stage Recommendation System Architecture

This folder contains architecture documentation for the yamyam-lab recommendation system.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Two-Stage Recommendation                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stage 1: Candidate Generation                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Multimodal Triplet Embedding Model                 │    │
│  │  - Input: Diner metadata (category, menu, reviews)  │    │
│  │  - Output: 128-d embedding                          │    │
│  │  - Retrieval: Top-K similar diners via dot product  │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                  │
│                          ▼                                  │
│  Stage 2: Reranking                                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  LightGBM / XGBoost Ranker                          │    │
│  │  - Input: User-diner features + candidates          │    │
│  │  - Output: Personalized ranking scores              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Documents

| Document | Description |
|----------|-------------|
| [multimodal_triplet.md](./multimodal_triplet.md) | Candidate generation model architecture |
| reranker.md | (TODO) Reranking model architecture |
| data_pipeline.md | (TODO) Data preprocessing pipeline |
| evaluation.md | (TODO) Evaluation metrics and methodology |

## Quick Links

- Training: `poetry run python -m yamyam_lab.train --model <model_name>`
- Reranking: `poetry run python -m yamyam_lab.rerank`
- Model configs: `config/models/`
