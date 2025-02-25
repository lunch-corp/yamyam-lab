import pandas as pd
from preprocess.diner_transform import CategoryProcessor, process_diner_data
from tools.google_drive import ensure_data_files

# 데이터 파일 경로 가져오기
data_paths = ensure_data_files()

review = pd.read_csv(data_paths["review"])
reviewer = pd.read_csv(data_paths["reviewer"])

review = pd.merge(review, reviewer, on="reviewer_id", how="left")

diner = pd.read_csv(data_paths["diner"])
diner_with_raw_category = pd.read_csv(data_paths["category"])


diner_idx_both_exist = set(review["diner_idx"].unique()) & set(
    diner["diner_idx"].unique()
)
review = review[review["diner_idx"].isin(diner_idx_both_exist)]
diner = diner[diner["diner_idx"].isin(diner_idx_both_exist)]

# step 3: replace diner_category with raw, unpreprocessed diner_category
# this is temporary preprocessing because preprocessed categories will be given
category_columns = [
    "diner_category_large",
    "diner_category_middle",
    "diner_category_small",
    "diner_category_detail",
]

if all(col in diner.columns for col in category_columns):
    # `category_columns`을 제외한 컬럼 목록 생성 (집합 연산으로 최적화)
    columns_exclude_category_columns = list(set(diner.columns) - set(category_columns))
    diner = diner[columns_exclude_category_columns]

# step 4: Logic for modifying restaurant categories
processor = CategoryProcessor(diner_with_raw_category)
diner_with_processd_category = processor.process_all().df

diner = pd.merge(
    left=diner,
    right=diner_with_processd_category,
    how="left",
    on="diner_idx",
)
converted_diner, validation_diner = process_diner_data(diner)


print("validation_diner", validation_diner)

print(converted_diner.info())

print(converted_diner.head())
