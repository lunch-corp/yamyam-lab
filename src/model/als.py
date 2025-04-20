import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix

class ALSPipeline:
    """
    Wrapper for the implicit ALS recommendation model.

    Parameters
    ----------
    als_params : dict or None, optional
        ALS hyperparameters. If None or missing keys, defaults are used:
        - alpha: 40
        - factors: 20
        - regularization: 0.1
        - iterations: 15
        - use_gpu: False
        - calculate_training_loss: True
    """
    def __init__(self, als_params: dict = None):
        # Use defaults if no parameters provided
        params = als_params or {}
        # Confidence scaling parameter
        self.alpha = params.get('alpha', 40)
        # Initialize ALS model with provided or default hyperparameters
        self.model = AlternatingLeastSquares(
            factors=params.get('factors', 20),
            regularization=params.get('regularization', 0.1),
            iterations=params.get('iterations', 15),
            use_gpu=params.get('use_gpu', False),
            calculate_training_loss=params.get('calculate_training_loss', True)
        )
        # Attributes to be populated after fitting
        self.user_cat = None
        self.item_cat = None
        self.Cui_csr = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, review_df: pd.DataFrame):
        """
        Train the ALS model on review data.

        Parameters
        ----------
        review_df : pd.DataFrame
            DataFrame containing columns ['reviewer_id', 'diner_idx', 'reviewer_review_score'].
        """
        # 1) Data preprocessing: drop NA and encode user/item IDs
        df2 = review_df[['reviewer_id', 'diner_idx', 'reviewer_review_score']].dropna()
        self.user_cat = df2['reviewer_id'].astype('category')
        self.item_cat = df2['diner_idx'].astype('category')
        rows = self.user_cat.cat.codes
        cols = self.item_cat.cat.codes
        data = df2['reviewer_review_score'].astype(np.float32)

        # 2) Build sparse user-item matrix (raw ratings)
        self.Cui_csr = coo_matrix(
            (data, (rows, cols)),
            shape=(self.user_cat.cat.categories.size, self.item_cat.cat.categories.size)
        ).tocsr()

        # 3) Scale confidence: c_ui = 1 + alpha * rating
        Cui_conf = self.Cui_csr * self.alpha
        Cui_conf.data += 1.0

        # 4) Fit ALS model on confidence-scaled matrix
        self.model.fit(Cui_conf)

        # 5) Store latent factors for predictions
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

    def predict_user_scores(self, user_index: int) -> np.ndarray:
        """
        Compute predicted scores for all items for a given user.

        Parameters
        ----------
        user_index : int
            Internal user index (0-based).

        Returns
        -------
        np.ndarray
            Array of predicted scores with shape (num_items,).
        """
        # Retrieve user's latent factor vector
        user_vec = self.user_factors[user_index]
        # Compute dot product with item latent factors
        return user_vec.dot(self.item_factors.T)

    def recommend_for_user(self, user_id, N: int = 10) -> list:
        """
        Generate top-N item recommendations for a given raw user ID.

        Parameters
        ----------
        user_id : raw user ID
        N : int, optional (default=10)
            Number of items to recommend.

        Returns
        -------
        list
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
    
    
    
if __name__ == "__main__":
    from src.tools.google_drive import ensure_data_files
    import pandas as pd

    # Ensure required data files are available
    data_paths = ensure_data_files()

    # Load data into Pandas DataFrames
    print(data_paths)
    review = pd.read_csv(data_paths["review"], index_col=0)
    
    als_pipeline = ALSPipeline()
    als_pipeline.fit(review_df=review)
    top10 = als_pipeline.recommend_for_user(user_id=256114348, N=10)