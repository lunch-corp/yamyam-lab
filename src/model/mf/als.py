from typing import List, Tuple

import numpy as np
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
        calculate_training_loss: bool = True,
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
        """
        return self.model.recommend(
            userid=user_ids,
            user_items=train_csr,
            filter_already_liked_items=filter_already_liked,
            N=topk,
        )

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
