from typing import Dict, List, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCollaborativeFiltering:
    def __init__(
        self,
        user_item_matrix: csr_matrix,
        item_embeddings: np.ndarray,
        user_mapping: Dict[int, int],
        item_mapping: Dict[int, int],
        diner_df: pd.DataFrame = None,
    ) -> None:
        """
        Initialize the ItemBasedCollaborativeFiltering with trained embeddings and user-item matrix.

        Args:
            user_item_matrix: CSR matrix of shape (n_users, n_items) with ratings/scores
            item_embeddings: Array of item embeddings from trained model (deprecated, use None)
            user_mapping: Dictionary mapping user_id to matrix row index
            item_mapping: Dictionary mapping restaurant_id to matrix column index
            diner_df: DataFrame containing diner information (optional, for hybrid)
        """
        self.user_item_matrix = user_item_matrix
        self.item_embeddings = item_embeddings
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.diner_df = diner_df

        # Create reverse mappings
        self.idx_to_user = {idx: user_id for user_id, idx in user_mapping.items()}
        self.idx_to_item = {idx: item_id for item_id, idx in item_mapping.items()}

        # Transpose for item-based operations (items x users)
        self.item_user_matrix = self.user_item_matrix.T

        # Prepare diner lookup if available
        if diner_df is not None:
            self.diner_info = diner_df.set_index("diner_idx").to_dict("index")
        else:
            self.diner_info = None

    def _get_item_idx(self, item_id: int) -> int:
        """Return item index or None if not found."""
        return self.item_mapping.get(item_id, None)

    def _get_user_idx(self, user_id: int) -> int:
        """Return user index or None if not found."""
        return self.user_mapping.get(user_id, None)

    def _jaccard_vector(self, target_idx: int) -> np.ndarray:
        """Compute Jaccard similarity between the target item and all others."""
        A = self.item_user_matrix
        if not sp.isspmatrix_csr(A):
            A = A.tocsr()

        # Binarize matrix (presence = 1)
        A_bin = A.copy()
        A_bin.data[:] = 1

        target_row = A_bin.getrow(target_idx)
        inter = A_bin.multiply(target_row).sum(axis=1).A1  # (n_items,)
        deg = A_bin.sum(axis=1).A1
        target_deg = target_row.sum()
        union = deg + target_deg - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = np.where(union > 0, inter / union, 0.0)
        sims[target_idx] = 0.0  # exclude self-similarity
        return sims

    def _cosine_vector(self, target_idx: int) -> np.ndarray:
        """Compute cosine similarity between target item and all others."""
        target_row = self.item_user_matrix.getrow(target_idx)
        sims = cosine_similarity(target_row, self.item_user_matrix).ravel()
        sims[target_idx] = 0.0
        return sims

    def _calculate_content_similarity(self, target_id: int, candidate_id: int) -> float:
        """
        Calculate simple content-based similarity.
        Simple version using only category information.
        """
        if self.diner_info is None:
            return 0.0

        if target_id not in self.diner_info or candidate_id not in self.diner_info:
            return 0.0

        target = self.diner_info[target_id]
        candidate = self.diner_info[candidate_id]

        # Simple category comparison
        if "diner_category_large" in target and "diner_category_large" in candidate:
            if target["diner_category_large"] == candidate["diner_category_large"]:
                return 1.0

        return 0.0

    def find_similar_items(
        self,
        target_item_id: int,
        top_k: int = 10,
        method: str = "cosine_matrix",
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Find similar items using item-based CF.

        Args:
            target_item_id: The item ID to find neighbors for.
            top_k: Number of similar items to return.
            method: "cosine_matrix" or "jaccard".

        Returns:
            List of dicts with 'item_id' and 'similarity_score' keys
        """
        target_idx = self._get_item_idx(target_item_id)
        if target_idx is None:
            return []

        if self.item_user_matrix.shape[1] == 0:
            return []

        if method not in {"cosine_matrix", "jaccard"}:
            method = "cosine_matrix"

        # Compute similarity vector
        sims = (
            self._cosine_vector(target_idx)
            if method == "cosine_matrix"
            else self._jaccard_vector(target_idx)
        )

        sims = np.nan_to_num(sims, nan=0.0, neginf=0.0, posinf=0.0)

        if sims.size == 0:
            return []

        # Get top-k item indices by similarity
        top_idx = np.argpartition(-sims, kth=min(top_k, sims.size - 1))[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        # Build result list
        results = []
        for idx in top_idx:
            item_id = self.idx_to_item.get(idx)
            if item_id is None:
                continue
            results.append(
                {"item_id": int(item_id), "similarity_score": float(sims[idx])}
            )
        return results

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        method: str = "cosine_matrix",
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Recommend items to a user based on item-based CF.

        Args:
            user_id: The ID of the user to recommend for.
            top_k: Number of items to recommend.
            method: "cosine_matrix" or "jaccard".

        Returns:
            List of dicts with 'item_id' and 'predicted_score' keys
        """
        user_idx = self._get_user_idx(user_id)
        if user_idx is None:
            return []

        user_row = self.user_item_matrix.getrow(user_idx)
        interacted_idx = user_row.indices
        interacted_val = user_row.data

        if interacted_idx.size == 0:
            return []

        n_items = self.item_user_matrix.shape[0]
        scores = np.zeros(n_items, dtype=np.float64)
        weights = np.zeros(n_items, dtype=np.float64)

        # Weighted sum of similarities for each interacted item
        for i_idx, r in zip(interacted_idx, interacted_val):
            sims = (
                self._jaccard_vector(i_idx)
                if method == "jaccard"
                else self._cosine_vector(i_idx)
            )
            scores += r * sims
            weights += np.abs(sims)

        denom = np.maximum(weights, 1e-12)
        preds = scores / denom
        preds[interacted_idx] = -np.inf  # exclude already seen items

        if not np.isfinite(preds).any():
            return []

        k = min(top_k, np.isfinite(preds).sum())
        top_idx = np.argpartition(-preds, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-preds[top_idx])]

        recs = []
        for idx in top_idx:
            item_id = self.idx_to_item.get(int(idx))
            if item_id is None:
                continue
            recs.append({"item_id": int(item_id), "predicted_score": float(preds[idx])})
        return recs

    def find_similar_items_hybrid(
        self,
        target_item_id: int,
        top_k: int = 10,
        cf_weight: float = 0.8,
        content_weight: float = 0.2,
        method: str = "cosine_matrix",
        normalize_weights: bool = True,
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Hybrid similarity: Combination of CF + Content.

        Args:
            target_item_id: Target restaurant ID
            top_k: Number of similar restaurants to return
            cf_weight: Weight for CF similarity (default 0.8)
            content_weight: Weight for content similarity (default 0.2)
            method: CF method ('cosine_matrix' or 'jaccard')
            normalize_weights: If True, normalize weights to sum to 1.0 (default True)

        Returns:
            List of similar restaurants sorted by hybrid score
        """
        # Normalize weights
        if normalize_weights:
            total = cf_weight + content_weight
            if total > 0:
                cf_weight /= total
                content_weight /= total

        # 1. Get CF-based similarities using existing method
        cf_results = self.find_similar_items(
            target_item_id=target_item_id,
            top_k=top_k * 3,  # Get more candidates
            method=method,
        )

        # 2. Calculate content similarities for each candidate
        hybrid_results = []

        for cf_item in cf_results:
            candidate_id = cf_item["item_id"]
            cf_score = cf_item["similarity_score"]

            # Content similarity
            content_score = (
                self._calculate_content_similarity(target_item_id, candidate_id)
                if content_weight > 0
                else 0.0
            )

            # Calculate hybrid score
            hybrid_score = cf_weight * cf_score + content_weight * content_score

            hybrid_results.append(
                {
                    "item_id": candidate_id,
                    "hybrid_score": hybrid_score,
                    "cf_score": cf_score,
                    "content_score": content_score,
                }
            )

        # 3. Re-sort by hybrid score
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return hybrid_results[:top_k]
