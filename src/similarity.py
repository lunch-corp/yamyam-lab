import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pytest

from src.model.classic_cf.item_based import ItemBasedCollaborativeFiltering

DATA_DIR = os.environ.get("DATA_DIR", "data")
DINER_PATH = os.path.join(DATA_DIR, "diner.csv")
DINER_CAT_PATH = os.path.join(DATA_DIR, "diner_category_raw.csv")
REVIEW_PATH = os.path.join(DATA_DIR, "review.csv")


@pytest.fixture(scope="session")
def raw_frames():
    assert os.path.exists(DINER_PATH), f"Not found: {DINER_PATH}"
    assert os.path.exists(DINER_CAT_PATH), f"Not found: {DINER_CAT_PATH}"
    assert os.path.exists(REVIEW_PATH), f"Not found: {REVIEW_PATH}"

    diner_df = pd.read_csv(DINER_PATH, low_memory=False)
    diner_cat_df = pd.read_csv(DINER_CAT_PATH, low_memory=False)
    review_df = pd.read_csv(REVIEW_PATH, low_memory=False)

    # 최소 컬럼 존재성 확인
    for col in ["diner_idx"]:
        assert col in diner_df.columns, f"missing column in diner.csv: {col}"
    for col in ["diner_idx"]:
        assert col in diner_cat_df.columns, f"missing column in diner_category_raw.csv: {col}"
    for col in ["reviewer_id", "diner_idx", "reviewer_review_score"]:
        assert col in review_df.columns, f"missing column in review.csv: {col}"

    return diner_df, diner_cat_df, review_df


@pytest.fixture(scope="session")
def prepared_data(raw_frames):
    diner_df, diner_cat_df, review_df = raw_frames

    # 가볍게 샘플링: 테스트 시간 단축을 위해 상위 N 사용자/아이템만 사용(실데이터지만 가볍게)
    # 그래도 "실제 데이터"에서 뽑기 때문에 파이프라인 변화 감지에 유효합니다.
    # 상위 사용자/아이템 선별
    top_users = (
        review_df["reviewer_id"].value_counts().head(2000).index.astype(int).tolist()
    )
    top_diners = (
        review_df["diner_idx"].value_counts().head(5000).index.astype(int).tolist()
    )
    df = review_df[
        review_df["reviewer_id"].isin(top_users) & review_df["diner_idx"].isin(top_diners)
    ].copy()

    # 결측/이상치 방어
    df = df.dropna(subset=["reviewer_id", "diner_idx", "reviewer_review_score"])
    # 평점 범위 클리핑 (혹시 모를 이상치)
    if df["reviewer_review_score"].dtype.kind in "if":
        df["reviewer_review_score"] = df["reviewer_review_score"].clip(lower=0)

    # 매핑 생성 (train_graph.py가 사용하는 'unique mapping id' 컨벤션과 일치)
    unique_users = np.sort(df["reviewer_id"].unique())
    unique_diners = np.sort(df["diner_idx"].unique())
    user_mapping = {int(uid): i for i, uid in enumerate(unique_users)}
    diner_mapping = {int(did): j for j, did in enumerate(unique_diners)}

    # CSR 행렬 생성 (users x items)
    rows = df["reviewer_id"].map(user_mapping).to_numpy()
    cols = df["diner_idx"].map(diner_mapping).to_numpy()
    vals = df["reviewer_review_score"].astype(float).to_numpy()

    n_users = len(user_mapping)
    n_items = len(diner_mapping)
    X = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

    # 컨텐츠 DF 준비 (hybrid에서 카테고리 사용)
    diner_meta = pd.merge(
        diner_df, diner_cat_df, on="diner_idx", how="left", suffixes=("", "_cat")
    )
    # 모델 내부에서 set_index("diner_idx")를 하므로 최소 필요한 컬럼 존재
    assert "diner_idx" in diner_meta.columns
    # 범주 컬럼이 없을 수도 있으니, 없으면 기본값 채워넣기
    if "diner_category_large" not in diner_meta.columns:
        diner_meta["diner_category_large"] = "unknown"

    # 모델 입력 준비
    return {
        "X": X,
        "user_mapping": user_mapping,
        "diner_mapping": diner_mapping,
        "diner_meta": diner_meta[["diner_idx", "diner_category_large"]].drop_duplicates(),
        "df": df,
    }


@pytest.fixture(scope="session")
def model_cf(prepared_data):
    X = prepared_data["X"]
    um = prepared_data["user_mapping"]
    im = prepared_data["diner_mapping"]
    diner_meta = prepared_data["diner_meta"]

    # 임베딩 없음 (기본 CF 전용)
    model = ItemBasedCollaborativeFiltering(
        user_item_matrix=X,
        item_embeddings=None,
        user_mapping=um,
        item_mapping=im,
        diner_df=diner_meta,
    )
    return model


def _popular_item_id_from_df(prepared_data):
    # target item을 데이터 기반으로 가장 인기 많은 아이템으로 선택 (테스트 안정성↑)
    counts = prepared_data["df"]["diner_idx"].value_counts()
    return int(counts.index[0])


def _active_user_id_from_df(prepared_data):
    counts = prepared_data["df"]["reviewer_id"].value_counts()
    return int(counts.index[0])


def test_find_similar_items_cosine_real_data(model_cf, prepared_data):
    target_item_id = _popular_item_id_from_df(prepared_data)
    res = model_cf.find_similar_items(target_item_id=target_item_id, top_k=10, method="cosine_matrix")
    assert isinstance(res, list) and len(res) > 0
    # 자기 자신 제외
    assert all(r["item_id"] != target_item_id for r in res)
    # 정렬
    sims = [r["similarity_score"] for r in res]
    assert all(sims[i] >= sims[i+1] - 1e-12 for i in range(len(sims)-1))


def test_find_similar_items_jaccard_real_data(model_cf, prepared_data):
    target_item_id = _popular_item_id_from_df(prepared_data)
    res = model_cf.find_similar_items(target_item_id=target_item_id, top_k=5, method="jaccard")
    assert isinstance(res, list) and len(res) > 0
    sims = [r["similarity_score"] for r in res]
    assert all(0.0 <= s <= 1.0 for s in sims)


def test_recommend_for_user_excludes_seen(model_cf, prepared_data):
    user_id = _active_user_id_from_df(prepared_data)
    seen = set(
        prepared_data["df"].loc[
            prepared_data["df"]["reviewer_id"] == user_id, "diner_idx"
        ].astype(int)
    )
    recs = model_cf.recommend_for_user(user_id=user_id, top_k=10, method="cosine_matrix")
    assert isinstance(recs, list)
    # 이미 본 아이템 제외
    assert all(r["item_id"] not in seen for r in recs)


def test_hybrid_with_content_only(model_cf, prepared_data):
    # 임베딩 없이 content만 섞는 하이브리드 (embedding_weight=0)
    target_item_id = _popular_item_id_from_df(prepared_data)
    res = model_cf.find_similar_items_hybrid(
        target_item_id=target_item_id,
        top_k=10,
        cf_weight=0.8,
        content_weight=0.2,
        embedding_weight=0.0,
        method="cosine_matrix",
        normalize_weights=True,
    )
    assert isinstance(res, list) and len(res) > 0
    # 키 존재/스코어 범위
    for r in res:
        for k in ("item_id", "hybrid_score", "cf_score", "content_score", "embedding_score"):
            assert k in r
        assert 0.0 <= r["content_score"] <= 1.0
        # 임베딩 0 가중치이므로 embedding_score는 계산되더라도 hybrid에 영향 X
    # 정렬
    hs = [r["hybrid_score"] for r in res]
    assert all(hs[i] >= hs[i+1] - 1e-12 for i in range(len(hs)-1))
    