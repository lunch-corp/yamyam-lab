import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")

# set cpu or cuda for default option
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device.type)


class TorchData(Dataset):
    def __init__(self, X, y):
        """
        X: torch.tensor
        y: torch.tensor
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_test_split_stratify(
    test_size,
    min_reviews,
    X_columns=["diner_idx", "reviewer_id"],
    y_columns=["reviewer_review_score"],
    random_state=42,
    stratify="reviewer_id",
):
    """
    test_size: ratio of test dataset
    min_reviews: minimum number of reviews for each reviewer
    X_columns: column names for model feature
    y_columns: column names for target value
    use_columns: columns to use in review data
    random_state: random seed for reproducibility
    stratify: column to stratify review data
    """
    # load data
    review_1 = pd.read_csv(os.path.join(DATA_PATH, "review_df_20241107_071929_yamyam_1.csv"))
    review_2 = pd.read_csv(os.path.join(DATA_PATH, "review_df_20241107_071929_yamyam_2.csv"))
    review = pd.concat([review_1, review_2], axis=0)[X_columns + y_columns]
    del review_1
    del review_2

    # store unique number of diner and reviewer
    diner_idxs = sorted(list(review["diner_idx"].unique()))
    reviewer_ids = sorted(list(review["reviewer_id"].unique()))

    num_diners = len(diner_idxs)
    num_reviewers = len(reviewer_ids)

    # mapping diner_idx and reviewer_id
    diner_mapping = {diner_idx: i for i, diner_idx in enumerate(diner_idxs)}
    reviewer_mapping = {reviewer_id: i for i, reviewer_id in enumerate(reviewer_ids)}
    review["diner_idx"] = review["diner_idx"].map(diner_mapping)
    review["reviewer_id"] = review["reviewer_id"].map(reviewer_mapping)

    # filter reviewer who wrote reviews more than min_reviews
    reviewer2review_cnt = review["reviewer_id"].value_counts().to_dict()
    reviewer_id_over = [reviewer_id for reviewer_id, cnt in reviewer2review_cnt.items() if cnt >= min_reviews]
    review_over = review[lambda x: x["reviewer_id"].isin(reviewer_id_over)]
    train, val = train_test_split(
        review_over, test_size=test_size, random_state=random_state, stratify=review_over[stratify]
    )
    return {
        "X_train": torch.tensor(train[X_columns].values),
        "y_train": torch.tensor(train[y_columns].values, dtype=torch.float32),
        "X_val": torch.tensor(val[X_columns].values),
        "y_val": torch.tensor(val[y_columns].values, dtype=torch.float32),
        "num_diners": num_diners,
        "num_users": num_reviewers,
        "diner_mapping": diner_mapping,
        "user_mapping": reviewer_mapping,
    }


def prepare_torch_dataloader(X_train, y_train, X_val, y_val, batch_size=128, random_state=42):
    seed = torch.Generator().manual_seed(random_state)

    train_dataset = TorchData(X_train, y_train)
    val_dataset = TorchData(X_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=seed)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=seed)
    return train_dataloader, val_dataloader
