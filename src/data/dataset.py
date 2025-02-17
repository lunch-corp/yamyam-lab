import pandas as pd
from preprocess.preprocess import preprocess_diner_data
from sklearn.preprocessing import LabelEncoder
from tools.google_drive import ensure_data_files


def load_test_dataset(reviewer_id: int) -> tuple[pd.DataFrame, list[str]]:
    """
    Load test dataset for inference
    params:
        reviewer_id: int

    return:
        test: pd.DataFrame
        already_reviewed: list[str]
    """
    # 필요한 데이터 다운로드 확인
    data_paths = ensure_data_files()

    # load data
    diner = pd.read_csv(data_paths["diner"]["file_path"])
    review = pd.read_csv(data_paths["review"]["file_path"]) 
    reviewer = pd.read_csv(data_paths["reviewer"]["file_path"])
    review = pd.merge(review, reviewer, on="reviewer_id", how="left")

    # label Encoder
    le = LabelEncoder()
    review["badge_grade"] = le.fit_transform(review["badge_grade"])

    diner = preprocess_diner_data(diner)

    # 사용자별 리뷰한 레스토랑 ID 목록 생성
    user_2_diner_df = review.groupby("reviewer_id").agg({"diner_idx": list})
    user_2_diner_map = dict(zip(user_2_diner_df.index, user_2_diner_df["diner_idx"]))

    # 레스토랑 후보군 리스트
    
    candidate_pool = diner["diner_idx"].unique().tolist()

    reviewed_diners = list(set(user_2_diner_map.get(reviewer_id, [])))
    candidates = [d for d in candidate_pool if d not in reviewed_diners]
    review = review[review["reviewer_id"] == reviewer_id].iloc[-1:]
    review = review.drop(columns=["diner_idx"])

    # Create test data
    test = pd.DataFrame({"reviewer_id": reviewer_id, "diner_idx": candidates})
    test = test.merge(diner, on="diner_idx", how="left")
    test = test.merge(review, on="reviewer_id", how="left")
    already_reviewed = user_2_diner_map.get(reviewer_id, [])

    return test, already_reviewed
