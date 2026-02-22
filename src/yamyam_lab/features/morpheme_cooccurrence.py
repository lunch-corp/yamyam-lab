"""Morpheme extraction and Jaccard-based positive pair generation.

This module tokenizes diner reviews into morpheme sets using Kiwi,
builds a binary morpheme occurrence matrix, and computes pairwise
Jaccard similarity within the same middle category to generate
positive pairs for the multimodal triplet embedding model.
"""

import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from yamyam_lab.tools.morpheme import tokenize_with_kiwi


def _init_worker():
    """Initializer for multiprocessing workers — creates a Kiwi instance per process."""
    from kiwipiepy import Kiwi

    import yamyam_lab.features.morpheme_cooccurrence as mod

    mod._worker_kiwi = Kiwi()


def _process_diner(args: Tuple[int, str, List[str]]) -> Tuple[int, Set[str]]:
    """Process a single diner's concatenated review text.

    Args:
        args: Tuple of (diner_idx, concatenated_text, included_tags).

    Returns:
        Tuple of (diner_idx, set of morphemes).
    """
    import yamyam_lab.features.morpheme_cooccurrence as mod

    diner_idx, text, included_tags = args
    tokens = tokenize_with_kiwi(mod._worker_kiwi, text, included_tags)
    return diner_idx, set(tokens)


def tokenize_reviews_parallel(
    review_df: pd.DataFrame,
    included_tags: List[str],
    n_processes: Optional[int] = None,
) -> Dict[int, Set[str]]:
    """Tokenize reviews per diner using multiprocessing.

    Groups reviews by diner_idx, concatenates review texts,
    and extracts morphemes in parallel.

    Args:
        review_df: DataFrame with columns [diner_idx, reviewer_review].
        included_tags: POS tags to keep (e.g., ["NNG", "NNP", "VA"]).
        n_processes: Number of worker processes. Defaults to cpu_count.

    Returns:
        Mapping from diner_idx to set of morphemes.
    """
    if n_processes is None:
        n_processes = mp.cpu_count()

    # Group reviews by diner and concatenate
    grouped = (
        review_df.groupby("diner_idx")["reviewer_review"]
        .apply(lambda texts: " ".join(texts.dropna().astype(str)))
        .reset_index()
    )

    tasks = [
        (int(row["diner_idx"]), row["reviewer_review"], included_tags)
        for _, row in grouped.iterrows()
    ]

    with mp.Pool(processes=n_processes, initializer=_init_worker) as pool:
        results = pool.map(_process_diner, tasks)

    return {diner_idx: morphemes for diner_idx, morphemes in results}


def build_morpheme_matrix(
    diner_morphemes: Dict[int, Set[str]],
    min_df: int = 3,
    max_df: float = 0.5,
    save_dir: Optional[str] = None,
) -> Tuple[sp.csr_matrix, List[str], Dict[int, int]]:
    """Build a binary sparse morpheme matrix for all diners.

    Args:
        diner_morphemes: Mapping from diner_idx to set of morphemes.
        min_df: Minimum document frequency (absolute count).
        max_df: Maximum document frequency as fraction of total diners.
        save_dir: If provided, save matrix and vocab to this directory.

    Returns:
        Tuple of (csr_matrix, vocab list, diner_idx_to_row_pos mapping).
    """
    diner_indices = sorted(diner_morphemes.keys())
    diner_idx_to_pos = {d: i for i, d in enumerate(diner_indices)}
    num_diners = len(diner_indices)

    # Collect all morphemes and count document frequency
    doc_freq: Dict[str, int] = {}
    for morpheme_set in diner_morphemes.values():
        for m in morpheme_set:
            doc_freq[m] = doc_freq.get(m, 0) + 1

    # Filter by min_df and max_df
    max_df_abs = int(num_diners * max_df)
    vocab = sorted(m for m, freq in doc_freq.items() if min_df <= freq <= max_df_abs)
    vocab_to_idx = {m: i for i, m in enumerate(vocab)}

    # Build binary sparse matrix
    mat = sp.lil_matrix((num_diners, len(vocab)), dtype=np.float32)
    for diner_idx in diner_indices:
        row = diner_idx_to_pos[diner_idx]
        for m in diner_morphemes[diner_idx]:
            if m in vocab_to_idx:
                mat[row, vocab_to_idx[m]] = 1.0

    csr = mat.tocsr()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(save_path / "morpheme_matrix.npz", csr)
        with open(save_path / "morpheme_vocab.pkl", "wb") as f:
            pickle.dump({"vocab": vocab, "diner_idx_to_pos": diner_idx_to_pos}, f)

    return csr, vocab, diner_idx_to_pos


def compute_jaccard_pairs(
    morpheme_matrix: sp.csr_matrix,
    category_df: pd.DataFrame,
    diner_idx_to_pos: Dict[int, int],
    threshold: float = 0.3,
) -> pd.DataFrame:
    """Compute Jaccard similarity pairs within same middle category.

    Args:
        morpheme_matrix: Binary sparse matrix (num_diners x vocab_size).
        category_df: DataFrame with columns [diner_idx, middle_category_id].
        diner_idx_to_pos: Mapping from diner_idx to matrix row position.
        threshold: Minimum Jaccard similarity to include a pair.

    Returns:
        DataFrame with columns [anchor_idx, positive_idx, jaccard_score],
        containing bidirectional pairs (A->B and B->A).
    """
    # Filter category_df to only diners in the matrix
    valid_diners = set(diner_idx_to_pos.keys())
    cat_filtered = category_df[category_df["diner_idx"].isin(valid_diners)].copy()

    anchors, positives, scores = [], [], []

    for _, group in cat_filtered.groupby("middle_category_id"):
        diner_list = group["diner_idx"].tolist()
        if len(diner_list) < 2:
            continue

        # Get row positions and extract submatrix
        positions = [diner_idx_to_pos[d] for d in diner_list]
        submatrix = morpheme_matrix[positions]

        # Compute pairwise Jaccard
        intersection = (submatrix @ submatrix.T).toarray()
        sizes = np.asarray(submatrix.sum(axis=1)).flatten()
        union = sizes[:, None] + sizes[None, :] - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard = np.where(union > 0, intersection / union, 0.0)

        # Collect upper-triangle pairs above threshold
        n = len(diner_list)
        for i in range(n):
            for j in range(i + 1, n):
                if jaccard[i, j] > threshold:
                    d_i, d_j = diner_list[i], diner_list[j]
                    score = float(jaccard[i, j])
                    # Bidirectional
                    anchors.extend([d_i, d_j])
                    positives.extend([d_j, d_i])
                    scores.extend([score, score])

    return pd.DataFrame(
        {"anchor_idx": anchors, "positive_idx": positives, "jaccard_score": scores}
    )


def analyze_threshold(
    morpheme_matrix: sp.csr_matrix,
    category_df: pd.DataFrame,
    diner_idx_to_pos: Dict[int, int],
    diner_df: Optional[pd.DataFrame] = None,
    thresholds: Optional[List[float]] = None,
) -> None:
    """Print pair counts and sample pairs at various Jaccard thresholds.

    Args:
        morpheme_matrix: Binary sparse matrix.
        category_df: DataFrame with [diner_idx, middle_category_id].
        diner_idx_to_pos: Mapping from diner_idx to row position.
        diner_df: Optional diner DataFrame with [diner_idx, diner_name] for display.
        thresholds: List of thresholds to evaluate.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    for t in thresholds:
        pairs_df = compute_jaccard_pairs(
            morpheme_matrix, category_df, diner_idx_to_pos, threshold=t
        )
        n_pairs = len(pairs_df) // 2  # bidirectional, so halve
        print(f"\nThreshold={t:.1f}: {n_pairs} unique pairs")

        if diner_df is not None and len(pairs_df) > 0:
            sample = pairs_df.head(6)
            name_map = dict(zip(diner_df["diner_idx"], diner_df["diner_name"]))
            for _, row in sample.iterrows():
                a_name = name_map.get(int(row["anchor_idx"]), "?")
                p_name = name_map.get(int(row["positive_idx"]), "?")
                print(f"  {a_name} <-> {p_name}  (jaccard={row['jaccard_score']:.3f})")
