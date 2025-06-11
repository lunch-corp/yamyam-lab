from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


class ALS:
    def __init__(
        self,
        alpha: float = 1.0,
        factors: int = 100,
        regularization: float = 0.01,
        iterations: int = 15,
        use_gpu: bool = False,
        diner_mapping: Dict[str, Any] = None,
        calculate_training_loss: bool = True,
        recommend_batch_size: int = 2000,
    ) -> None:
        """
        Wrapper for the implicit ALS recommendation model using default parameter values from implicit library.

        Args:
            alpha (float): Confidence level when defining C_{ui} = 1 + alpha * r_{ui}
            factors (int): Dimension of user/item embedding vector.
            regularization (float): Regularization parameter to prevent overfitting.
            iterations (float): Number of iterations while training.
            use_gpu (bool): Whether to use gpu or not.
            calculate_training_loss (bool): Whether to calculate loss or not while training.
        """
        # Confidence scaling parameter
        self.alpha = alpha
        # Initialize ALS model with provided or default hyperparameters
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=use_gpu,
            calculate_training_loss=calculate_training_loss,
        )
        self.recommend_batch_size = recommend_batch_size
        self.diner_mapping = diner_mapping
        # Attributes to be populated after fitting
        self.user_cat = None
        self.item_cat = None
        self.Cui_csr = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, train_Cui_csr: csr_matrix) -> None:
        """
        Train the ALS model on csr converted review data.

        Args:
            train_Cui_csr (csr_matrix): Csr matrix using train dataset.
        """
        # Scale confidence: C_ui = 1 + alpha * rating
        Cui_conf = train_Cui_csr * self.alpha
        Cui_conf.data += 1.0

        # Fit ALS model on confidence-scaled matrix
        self.model.fit(Cui_conf)

        # Store latent factors for predictions
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

    def recommend(
        self,
        user_ids: NDArray,
        train_csr: csr_matrix,
        topk: int,
        filter_already_liked: bool = True,
    ) -> Tuple[NDArray, NDArray]:
        """
        Generate recommendation using method implemented in implicit library.

        Args:
            user_ids (NDArray): List of user_ids to generate recommendations.
            train_csr (csr_matrix): Training csr matrix to filter already liked items.
            filter_already_liked (bool): If set True, do not recommend items already liked in train dataset.

        Returns (Tuple[NDArray, NDArray]):
            Tuple of top_k_ids and top_k_values in numpy array.
        """
        return self.model.recommend(
            userid=user_ids,
            user_items=train_csr,
            filter_already_liked_items=filter_already_liked,
            N=topk,
        )

    def generate_candidates_for_each_user(
        self, top_k_value: int, train_csr: csr_matrix, filter_already_liked: bool = True
    ) -> pd.DataFrame:
        """
        Generate top_k candidates for all users.

        Args:
            top_k_value (int): Top k value which is number of candidates to be passed to ranker model.
            train_csr (csr_matrix): Train csr_matrix which will be used when filtering already liked items.
            filter_already_liked (bool): Whether filtering already liked item in train dataset or not.

        Returns (pd.DataFrame):
            Dataframe containing user_id, diner_id, associated score between user_id and diner_id.
        """
        num_users, D = self.model.user_factors.shape
        all_user_ids = np.arange(num_users)
        res = np.empty((0, 3))  # user_id, diner_id, score -> total 3 columns
        num_diners = len(self.diner_mapping)

        # Create reverse mapping
        diner_mapping_reverse = {v: k for k, v in self.diner_mapping.items()}

        for start in range(0, num_users, self.recommend_batch_size):
            user_ids = all_user_ids[start : start + self.recommend_batch_size]

            top_k_ids, top_k_values = self.recommend(
                user_ids=user_ids,
                train_csr=train_csr[user_ids],
                filter_already_liked=filter_already_liked,
                topk=top_k_value,
            )

            # Map diner IDs back to original IDs using vectorized operation
            top_k_ids_original = np.vectorize(diner_mapping_reverse.get)(top_k_ids)
            candi = np.concatenate(
                (
                    np.repeat(user_ids + num_diners, top_k_value).reshape(-1, 1),
                    top_k_ids_original.reshape(-1, 1),
                    top_k_values.reshape(-1, 1),
                ),
                axis=1,
            )

            res = np.concatenate((res, candi), axis=0)

        dtypes = {"user_id": np.int64, "diner_id": np.int64, "score": np.float64}
        res = pd.DataFrame(res, columns=list(dtypes.keys())).astype(dtypes)
        return res

    def predict_user_scores(self, user_index: int) -> NDArray:
        """
        Compute predicted scores for all items for a given user.

        Args:
            user_index (int): Internal user index (0-based).

        Returns (np.ndarray):
            Array of predicted scores with shape (num_items,).
        """
        # Retrieve user's latent factor vector
        user_vec = self.user_factors[user_index]
        # Compute dot product with item latent factors
        return user_vec.dot(self.item_factors.T)

    def recommend_for_user(self, user_id, N: int = 10) -> List[int]:
        """
        Generate top-N item recommendations for a given raw user ID.

        Args:
            user_id (int): Raw user ID
            N (int): Number of items to recommend.

        Returns (List[int]):
            List of recommended raw item IDs.
        """
        # 1) Map raw user ID to internal index
        try:
            uidx = self.user_cat.cat.categories.get_loc(user_id)
        except KeyError:
            raise ValueError(f"Unknown user_id: {user_id}")

        # 2) Compute scores for all items
        scores = self.predict_user_scores(uidx)

        # 3) Exclude already rated items
        seen = set(self.Cui_csr[uidx].indices)
        scores[list(seen)] = -np.inf

        # 4) Select top-N indices
        top_idx = np.argpartition(-scores, N)[:N]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        # 5) Convert internal indices back to raw item IDs
        return list(self.item_cat.cat.categories[top_idx])
