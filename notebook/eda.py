# %%
import pandas as pd

review = pd.read_csv("../data/review_df_20241107_071929_yamyam_1.csv", index_col=0)
diner = pd.read_csv("../data/diner_df_20241107_071929_yamyam.csv", index_col=0)

# %%
review.head()
# %%
diner.head()
# %%
diner["diner_category_large"].unique()
# %%
diner["diner_category_small"].unique()
# %%
diner.columns
# %%
diner["diner_tag"]
# %%
diner["diner_address_constituency"]
# %%
diner["diner_address"]
# %%
diner["diner_open_time"]
# %%
review.columns
# %%
[
    "reviewer_review_cnt",
    "reviewer_avg",
    "reviewer_review_score",,
    "diner_idx",
    "reviewer_collected_review_cnt",
    "diner_review_cnt",
    "diner_blog_review_cnt",
    "diner_review_avg",
    "diner_review_tags",
    "diner_lat",
    "diner_lon",
    "all_review_cnt",
    "real_good_review_percent",
    "real_bad_review_percent",
]
