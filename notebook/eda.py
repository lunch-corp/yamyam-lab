# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
# Load data (same as your current implementation)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

# load data
review_1 = pd.read_csv(os.path.join(DATA_PATH, "review/review_df_20241122_part_1.csv"))
review_2 = pd.read_csv(os.path.join(DATA_PATH, "review/review_df_20241122_part_2.csv"))
review_3 = pd.read_csv(os.path.join(DATA_PATH, "review/review_df_20241122_part_3.csv"))
review = pd.concat([review_1, review_2, review_3], axis=0)

# %%
sns.countplot(x="reviewer_review_score", data=review)
plt.show()
# %%
review.head()
# %%
diner = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241122_yamyam.csv"))
# %%
diner.head()
# %%
sns.histplot(diner["real_good_review_cnt"], bins=100, kde=True)
plt.show()
# %%
