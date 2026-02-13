# Multimodal Triplet Embedding Model

This document describes the architecture of the multimodal triplet embedding model used for candidate generation in the two-stage recommendation system.

## Overview

The multimodal triplet embedding model creates 128-dimensional embeddings for restaurants (diners) such that semantically similar restaurants have high dot-product similarity. This enables fast approximate nearest neighbor search for the "Similar Diners" feature.

**Goal**: Distinguish restaurants by style/quality, not just category.
- Shake Shack → premium American burger joints ✓
- Shake Shack → McDonald's ✗ (same category but different tier)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT FEATURES                         │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Category   │     Menu     │  Diner Name  │     Price      │
│   (3 IDs)    │   (768-d)    │   (768-d)    │   (3 floats)   │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘
       │              │              │                │
       ▼              ▼              ▼                ▼
┌──────────────┬──────────────┬──────────────┬────────────────┐
│  Category    │    Menu      │ Diner Name   │    Price       │
│  Encoder     │   Encoder    │   Encoder    │   Encoder      │
│              │              │              │                │
│  Embedding   │    MLP       │    MLP       │     MLP        │
│  + MLP       │  (768→256)   │  (768→64)    │   (3→32)       │
│              │              │              │                │
│  → 128-d     │   → 256-d    │   → 64-d     │    → 32-d      │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘
       │              │              │                │
       └──────────────┴──────────────┴────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Attention   │
                    │   Fusion     │
                    │  (4 heads)   │
                    │   → 480-d    │
                    └──────┬───────┘
                           │
                    ┌──────▼──────┐
                    │  Final MLP   │
                    │  (480→256    │
                    │   →128)      │
                    │  + L2 Norm   │
                    └──────┬───────┘
                           │
                    ┌──────▼──────┐
                    │  Embedding   │
                    │   (128-d)    │
                    │  unit norm   │
                    └──────────────┘
```

### Encoder Details

#### 1. Category Encoder
Encodes hierarchical category information (large → middle → small).

| Component | Input | Output |
|-----------|-------|--------|
| Large category embedding | ID (0-25) | 32-d |
| Middle category embedding | ID (0-297) | 48-d |
| Small category embedding | ID (0-587) | 64-d |
| Concatenation | 32+48+64 | 144-d |
| MLP (2 layers) | 144-d | 128-d |

#### 2. Menu Encoder
Encodes aggregated menu item text using pre-trained Korean BERT.

| Component | Input | Output |
|-----------|-------|--------|
| KoBERT (klue/bert-base) | Menu text tokens | 768-d |
| Mean pooling | Token embeddings | 768-d |
| MLP (2 layers) | 768-d | 256-d |

Note: KoBERT weights are **frozen** during training to prevent overfitting.

#### 3. Diner Name Encoder
Encodes diner (restaurant) name using pre-trained Korean BERT.

| Component | Input | Output |
|-----------|-------|--------|
| KoBERT (klue/bert-base) | Diner name tokens | 768-d |
| Mean pooling | Token embeddings | 768-d |
| MLP (2 layers) | 768-d | 64-d |

Note: KoBERT weights are **frozen** during training. The diner name captures brand identity and style information (e.g., "Shake Shack" vs "McDonald's").

#### 4. Price Encoder
Encodes price statistics derived from menu items.

| Feature | Description |
|---------|-------------|
| avg_price | Log-normalized average price |
| min_price | Log-normalized minimum price |
| max_price | Log-normalized maximum price |

MLP: 3 → 16 → 32 (2 layers with ReLU + Dropout)

#### 5. Attention Fusion
Multi-head attention over the 4 encoder outputs to learn which modalities are most important.

- Input: Concatenated encodings (128 + 256 + 64 + 32 = 480-d)
- Heads: 4
- Output: 480-d with residual connection + LayerNorm

#### 6. Final Projection
Projects fused features to final embedding space.

- MLP: 480 → 256 → 128
- L2 normalization to unit sphere

**Output**: 128-dimensional L2-normalized embedding where `dot(a, b) = cosine_similarity(a, b)`

---

## Loss Function

### Triplet Margin Loss with Category Regularization

```
Total Loss = Triplet Loss + λ × Category Loss
```

where λ = 0.1 (category_weight)

#### Triplet Margin Loss

```python
triplet_loss = max(0, margin + sim(anchor, negative) - sim(anchor, positive))
```

- **margin**: 0.5
- **similarity**: dot product (since embeddings are L2-normalized)
- **anchor**: Target diner embedding
- **positive**: Similar diner (co-reviewed or same category)
- **negative**: Hard negative (same category but different diner)

#### Category Consistency Loss

Cross-entropy loss predicting the middle category from the embedding. This ensures embeddings respect category structure while learning finer distinctions.

```python
category_loss = CrossEntropy(Linear(embedding), middle_category_id)
```

---

## Dataset

### Training Pairs

Two sources of positive pairs:

| Source | Description | Confidence |
|--------|-------------|------------|
| **Co-review pairs** | Diners reviewed by same users (≥2 common users) | High |
| **Category pairs** | Diners in same small/middle category | Medium |

### Hard Negative Mining

For each anchor, negatives are sampled with the following strategy:

| Type | Count | Description |
|------|-------|-------------|
| Hard negatives | 5 | Same large category as anchor |
| Semi-hard negatives | 3 | Nearby categories |
| Random negatives | 2 | Uniformly sampled |

This forces the model to distinguish quality/style within the same category (e.g., Shake Shack vs McDonald's).

### Data Files

| File | Description |
|------|-------------|
| `diner_features.parquet` | Preprocessed features for all diners |
| `training_pairs.parquet` | Positive training pairs |
| `val_pairs.parquet` | Validation pairs |
| `test_pairs.parquet` | Test pairs |
| `category_mapping.parquet` | Category ID mappings |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 128 |
| Batch size | 256 triplets |
| Learning rate | 0.001 |
| Optimizer | AdamW (weight_decay=0.01) |
| Gradient clipping | 1.0 |
| Triplet margin | 0.5 |
| Category loss weight | 0.1 |
| Early stopping patience | 5 epochs |

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Recall@K | % of positive pairs where positive is in top-K | >60% at K=10 |
| MRR | Mean Reciprocal Rank | >0.35 |
| Category Coherence | % of top-10 neighbors with same category | 50-70% |

---

## Usage

### Training

```bash
# Prepare data
poetry run python scripts/prepare_diner_embedding_data.py --local_data_dir data/

# Train model
poetry run python -m yamyam_lab.train --model multimodal_triplet --epochs 50 --device cuda
```

### Inference

```python
from yamyam_lab.model.embedding.multimodal_triplet import Model

# Load trained model
model = Model.load("result/multimodal_triplet/best_model.pt")

# Get embedding for a diner
embedding = model.encode(diner_features)  # (1, 128)

# Find similar diners via dot product
similarities = embedding @ all_embeddings.T  # (1, num_diners)
top_k_indices = similarities.argsort(descending=True)[:10]
```

---

## File References

| File | Description |
|------|-------------|
| `src/yamyam_lab/model/embedding/multimodal_triplet.py` | Main model class |
| `src/yamyam_lab/model/embedding/encoders.py` | Encoder modules |
| `src/yamyam_lab/loss/triplet.py` | Triplet loss functions |
| `src/yamyam_lab/data/multimodal_triplet.py` | Dataset and DataLoader |
| `src/yamyam_lab/engine/multimodal_triplet_trainer.py` | Training logic |
| `config/models/embedding/multimodal_triplet.yaml` | Hyperparameters |
