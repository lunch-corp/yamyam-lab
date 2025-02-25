try:
    import os
    import sys

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
    )

except ModuleNotFoundError:
    raise Exception("No module found")

from preprocess.preprocess import train_test_split_stratify


def test_train_test_split_stratify():
    data = train_test_split_stratify(
        test_size=0.2,
        min_reviews=10,
        random_state=42,
        stratify="reviewer_id",
    )

    assert data["X_train"].shape[0] > 10000
    assert data["X_val"].shape[0] > 10000
    assert data["y_train"].shape[0] > 10000
    assert data["y_val"].shape[0] > 10000
    assert data["num_diners"] > 0
    assert data["num_users"] > 0
    assert data["diner_mapping"] is not None
    assert data["user_mapping"] is not None


def test_train_test_split_stratify_rank():
    data = train_test_split_stratify(
        test_size=0.2,
        min_reviews=10,
        random_state=42,
        stratify="reviewer_id",
        is_rank=True,
        diner_engineered_feature_names=["all_review_cnt"],
    )

    assert data["X_train"].shape[0] > 10000
    assert data["X_val"].shape[0] > 10000
    assert data["y_train"].shape[0] > 10000
    assert data["y_val"].shape[0] > 10000
    assert data["X_train"].shape[0] == data["y_train"].shape[0]
    assert data["X_val"].shape[0] == data["y_val"].shape[0]
    assert data["num_diners"] > 0
    assert data["num_users"] > 0
    assert data["diner_mapping"] is not None
    assert data["user_mapping"] is not None
