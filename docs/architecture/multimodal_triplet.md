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
┌──────────────────────────────────────────────────────────────────────┐
│                          INPUT FEATURES                              │
├──────────┬──────────┬──────────┬──────────┬──────────────────────────┤
│ Category │   Menu   │  Diner   │  Price   │      Review Text         │
│ (2 IDs)  │ (768-d)  │  Name    │(3 floats)│        (768-d)           │
│          │          │ (768-d)  │          │                          │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴────────────┬────────────┘
     │          │          │          │                   │
     ▼          ▼          ▼          ▼                   ▼
┌──────────┬──────────┬──────────┬──────────┬──────────────────────────┐
│ Category │  Menu    │  Diner   │  Price   │     Review Text          │
│ Encoder  │ Encoder  │  Name    │ Encoder  │      Encoder             │
│          │          │ Encoder  │          │                          │
│ Emb+MLP  │   MLP    │   MLP    │   MLP    │        MLP               │
│  → 128-d │ → 256-d  │  → 64-d  │  → 32-d  │      → 128-d            │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴────────────┬────────────┘
     │          │          │          │                   │
     └──────────┴──────────┴──────────┴───────────────────┘
                              │
                       ┌──────▼──────┐
                       │  Attention   │
                       │   Fusion     │
                       │  (4 heads)   │
                       │   → 608-d    │
                       └──────┬───────┘
                              │
                       ┌──────▼──────┐
                       │  Final MLP   │
                       │  (608→256    │
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
Encodes hierarchical category information (large → middle).

| Component | Input | Output |
|-----------|-------|--------|
| Large category embedding | ID (0-25) | 32-d |
| Middle category embedding | ID (0-297) | 48-d |
| Concatenation | 32+48 | 80-d |
| MLP (2 layers) | 80-d | 128-d |

Note: Small category was removed due to high NA rates in the source data.

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

#### 5. Review Text Encoder
Encodes aggregated review text using pre-trained Korean BERT.

| Component | Input | Output |
|-----------|-------|--------|
| KoBERT (klue/bert-base) | Review text tokens | 768-d |
| Mean pooling | Token embeddings | 768-d |
| MLP (2 layers) | 768-d | 128-d |

Note: KoBERT weights are **frozen** during training. Review text captures diner atmosphere, food quality, and customer sentiment.

#### 6. Attention Fusion
Multi-head attention over the 5 encoder outputs to learn which modalities are most important.

- Input: Concatenated encodings (128 + 256 + 64 + 32 + 128 = 608-d)
- Heads: 4
- Output: 608-d with residual connection + LayerNorm

#### 7. Final Projection
Projects fused features to final embedding space.

- MLP: 608 → 256 → 128
- L2 normalization to unit sphere

**Output**: 128-dimensional L2-normalized embedding where `dot(a, b) = cosine_similarity(a, b)`

---

## Loss Function

### In-Batch InfoNCE Loss (CLIP-style)

The model uses in-batch contrastive learning where other samples in the batch serve as negatives.

```
sim_matrix = anchor_emb @ positive_emb.T / temperature   # (B, B)
loss = CrossEntropy(sim_matrix, labels=[0, 1, 2, ..., B-1])
```

- **temperature**: 0.07
- **labels**: Diagonal entries are the correct positive for each anchor
- **positive masking**: Known positive pairs (from morpheme Jaccard) are masked to `-inf` so they don't act as false negatives in the denominator

This replaces explicit negative mining with implicit in-batch negatives, scaling naturally with batch size (512).

---

## Dataset

### Positive Pair Generation (Morpheme Jaccard)

Positive pairs are defined by review morpheme similarity within the same middle category:

1. **Morpheme extraction**: Tokenize all reviews per diner using Kiwi (Korean NLP), keeping nouns (NNG, NNP) and adjectives (VA)
2. **Vocabulary filtering**: Remove morphemes appearing in <3 diners or >50% of diners
3. **Binary morpheme matrix**: Sparse (num_diners × vocab_size) matrix
4. **Jaccard similarity**: Computed within each middle category group
5. **Threshold**: Pairs with Jaccard > M are positive pairs (default M=0.3)

This approach ensures positive pairs share similar review language (food style, atmosphere) rather than just belonging to the same category.

### Negative Sampling

Negatives are sampled **in-batch**: for a batch of B anchor-positive pairs, each anchor treats the B-1 other positives as negatives. Known positive pairs are masked out to avoid false negatives.

### Data Preparation

```bash
# Step 1: Generate diner features (embeddings, category IDs, prices)
poetry run python scripts/prepare_diner_embedding_data.py --local_data_dir data/

# Step 2: Generate morpheme-based positive pairs
poetry run python scripts/prepare_morpheme_pairs.py --local_data_dir data/ --jaccard_threshold 0.3
```

### Data Files

| File | Description |
|------|-------------|
| `diner_features.parquet` | Preprocessed features for all diners |
| `training_pairs.parquet` | Positive training pairs (morpheme Jaccard) |
| `val_pairs.parquet` | Validation pairs |
| `test_pairs.parquet` | Test pairs |
| `category_mapping.parquet` | Category ID mappings |
| `morpheme_matrix.npz` | Binary sparse morpheme matrix |
| `morpheme_vocab.pkl` | Morpheme vocabulary and diner-to-row mapping |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 128 |
| Batch size | 512 |
| Learning rate | 0.001 |
| Optimizer | AdamW (weight_decay=1e-5) |
| Gradient clipping | 1.0 |
| Loss | In-batch InfoNCE |
| Temperature | 0.07 |
| Early stopping patience | 10 epochs |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Recall@K | % of positive pairs where positive is in top-K |
| MRR | Mean Reciprocal Rank |

---

## Usage

### Training

```bash
# Prepare data
poetry run python scripts/prepare_diner_embedding_data.py --local_data_dir data/
poetry run python scripts/prepare_morpheme_pairs.py --local_data_dir data/ --jaccard_threshold 0.3

# Train model
poetry run python -m yamyam_lab.train --model multimodal_triplet --epochs 100 --batch_size 512
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
| `src/yamyam_lab/loss/infonce.py` | In-batch InfoNCE loss + positive mask builder |
| `src/yamyam_lab/loss/triplet.py` | Triplet loss functions (legacy) |
| `src/yamyam_lab/data/multimodal_triplet.py` | Dataset and DataLoader |
| `src/yamyam_lab/engine/multimodal_triplet_trainer.py` | Training logic |
| `src/yamyam_lab/features/morpheme_cooccurrence.py` | Morpheme extraction + Jaccard pairs |
| `config/models/embedding/multimodal_triplet.yaml` | Hyperparameters |
| `scripts/prepare_diner_embedding_data.py` | Feature preprocessing script |
| `scripts/prepare_morpheme_pairs.py` | Morpheme pair generation script |
