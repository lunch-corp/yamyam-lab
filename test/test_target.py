# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
# Load data (same as your current implementation)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

# Load data
review_data_paths = glob.glob(os.path.join(DATA_PATH, "review", "*.csv"))
diner = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241211_yamyam.csv"))
# %%
review = pd.DataFrame()
for review_data_path in review_data_paths:
    review = pd.concat([review, pd.read_csv(review_data_path)], axis=0)

# %%
review.head()
# %%
review = pd.merge(review, diner, on="diner_idx", how="inner")
# %%

# 베이지안 평균
min_reviews = 100
review["target"] = (
    min_reviews * review["reviewer_avg"] + review["reviewer_review_score"]
) / (min_reviews + review["all_review_cnt"])

review["target"] = np.where(review["target"] > 4, 1, 0)

# %%
review["target"] = np.where(
    (review["real_good_review_percent"] > review["real_bad_review_percent"])
    & (review["reviewer_review_score"] - review["reviewer_avg"] >= 0.5),
    1,
    0,
)
