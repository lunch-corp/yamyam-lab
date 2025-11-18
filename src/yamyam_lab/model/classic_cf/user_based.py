from typing import Dict, List, Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCollaborativeFiltering:
    def __init__(
        self,
        user_item_matrix: csr_matrix,
        user_mapping: Dict[int, int],
        item_mapping: Dict[int, int],
    ) -> None:
        """
        Initialize the UserBasedCollaborativeFiltering with trained embeddings and user-item matrix.

        Args:
            user_item_matrix: CSR matrix of shape (n_users, n_items) with ratings/scores
            user_embeddings: Array of user embeddings from trained model
            user_mapping: Dictionary mapping user_id to matrix row index
            item_mapping: Dictionary mapping restaurant_id to matrix column index
        """
        self.user_item_matrix = user_item_matrix
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping

        # Create reverse mappings
        self.idx_to_user = {idx: user_id for user_id, idx in user_mapping.items()}
        self.idx_to_item = {idx: item_id for item_id, idx in item_mapping.items()}

    def _create_cold_user_vector(
        self, liked_item_ids: List[int], scores_of_liked_items: List[int]
    ) -> np.ndarray:
        """
        Create sparse vector representation for a cold user based on their liked items and scores.

        Args:
            liked_item_ids: List of item IDs that the cold user liked
            scores_of_liked_items: List of scores corresponding to the liked items

        Returns:
            numpy array representing the cold user's preferences
        """
        cold_vector = np.zeros(len(self.item_mapping))
        for item_id, score in zip(liked_item_ids, scores_of_liked_items):
            if item_id in self.item_mapping:
                item_idx = self.item_mapping[item_id]
                cold_vector[item_idx] = score  # Use the score as the value

        return csr_matrix(cold_vector.reshape(1, -1))

    def find_similar_users(
        self,
        liked_item_ids: List[int],
        scores_of_liked_items: List[int],
        method: str = "cosine_matrix",
    ) -> Union[int, str]:
        """
        Find warm users most similar to cold user based on restaurant preferences.

        Args:
            liked_item_ids: List of item IDs selected by cold user
            scores_of_liked_items: List of scores for the liked items
            method: Similarity method to use ('jaccard' or 'cosine_matrix')

        Returns:
            User ID of the most similar user (int for cosine_matrix, str for jaccard)

        Raises:
            ValueError: If unknown method is specified
        """
        if method == "jaccard":
            return self._jaccard_similarity(liked_item_ids)
        elif method == "cosine_matrix":
            return self._cosine_matrix_similarity(liked_item_ids, scores_of_liked_items)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _cosine_matrix_similarity(
        self, liked_item_ids: List[int], scores_of_liked_items: List[int]
    ) -> int:
        """
        Find most similar user using cosine similarity with the CSR matrix directly.

        Uses cosine similarity to compare the cold user's preference vector with all warm users.
        When similarity scores are equal, prefers users with fewer liked items.

        Args:
            liked_item_ids: List of item IDs that the cold user liked
            scores_of_liked_items: List of scores corresponding to the liked items

        Returns:
            Index of the most similar user in the user-item matrix
        """
        cold_vector = self._create_cold_user_vector(
            liked_item_ids, scores_of_liked_items
        )

        # Compute cosine similarity between cold user vector and all warm users
        similarities_matrix = cosine_similarity(cold_vector, self.user_item_matrix)
        similarities_array = similarities_matrix.flatten()

        # Get number of liked items for each user (number of non-zero entries)
        num_liked_items = np.array(
            [
                self.user_item_matrix[i].nnz
                for i in range(self.user_item_matrix.shape[0])
            ]
        )

        # Create list of [user_idx, similarity, num_liked_items] for sorting
        sort_criteria = [
            [i, similarities_array[i], num_liked_items[i]]
            for i in range(len(similarities_array))
        ]

        # Sorting criteria:
        # First sort by cosine similarity with descending order
        # When cosine similarity is equal, sort by number of liked items with ascending order
        # We regard that users who have smaller number of liked items are more close to cold user,
        # when they have identical cosine similarity score.
        sorted_users = sorted(sort_criteria, key=lambda x: (-x[1], x[2]))

        # top 1 user
        return self.idx_to_user[sorted_users[0][0]]

    def _jaccard_similarity(self, liked_item_ids: List[int]) -> int:
        """
        Find most similar user using Jaccard similarity with vectorized operations.

        Uses Jaccard similarity (intersection over union) to compare binary preference vectors.
        Note: This method does not use rating scores, only binary preferences.

        Args:
            liked_item_ids: List of item IDs that the cold user liked

        Returns:
            User ID of the most similar user
        """
        # Create cold user binary vector
        cold_vector = self._create_cold_user_vector(
            liked_item_ids, [1] * len(liked_item_ids)
        )

        # Keep sparse matrix operations - convert to binary CSR matrix
        binary_matrix = (self.user_item_matrix > 0).astype(np.int8)

        # Vectorized intersection: sparse matrix dot product
        intersections = binary_matrix.dot(cold_vector.astype(int))

        # Vectorized union: |A| + |B| - |A ∩ B|
        cold_items_count = np.sum(cold_vector)
        warm_items_counts = np.array(binary_matrix.sum(axis=1)).flatten()
        unions = warm_items_counts + cold_items_count - intersections

        # Compute Jaccard similarities (avoid division by zero)
        jaccard_similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions != 0,
        )

        # Find best user: highest similarity, then highest intersection as tiebreaker
        best_idx = np.lexsort((intersections, jaccard_similarities))[-1]

        # Convert matrix index back to user_id
        return self.idx_to_user[best_idx]
