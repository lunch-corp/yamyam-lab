# Two-Stage Search System

이 문서는 검색 쿼리 기반 Two-Stage 검색 시스템의 설계 플랜을 기술한다.
기존 추천 파이프라인(user → similar diners)을 확장하여 **검색 파이프라인(query → ranked results)**을 구현한다.

---

## Overview

### 추천 vs 검색의 차이

| 구분 | 기존 추천 | 검색 |
|------|-----------|------|
| 입력 | `reviewer_id` | `search_query` |
| Positive 신호 | 리뷰 작성 | 클릭 (impression → click) |
| Stage 1 쿼리 | 유저 임베딩 | 쿼리 텍스트 임베딩 |
| Stage 2 피처 | 유저/식당 피처 | 쿼리-식당 관련도 + 식당 피처 |
| 그룹 단위 | `reviewer_id` | `query_id` |

실제 검색 로그가 없으므로 **기존 리뷰 데이터에서 가상 검색 로그를 생성**하는 것이 핵심 과제다.

---

## 실제 데이터 스키마

### 보유 데이터

| 파일 | 행 수 | 주요 컬럼 |
|------|-------|-----------|
| `data/review.csv` | 1,865,556 | `review_id`, `diner_idx`, `reviewer_id`, `reviewer_review_score`, `reviewer_review_date`, `reviewer_review` |
| `data/diner.csv` | 124,111 | `diner_idx`, `diner_name`, `diner_road_address`, `diner_lat`, `diner_lon`, `diner_review_cnt`, `bayesian_score` |
| `data/reviewer.csv` | 554,884 | `reviewer_id`, `reviewer_user_name`, `reviewer_review_cnt`, `reviewer_avg`, `badge_level`, `badge_grade` |
| `data/diner_category.csv` | 124,111 | `diner_idx`, `diner_category_large`, `diner_category_middle`, `diner_category_small` |
| `data/menu_df.csv` | 1,581,773 | `diner_idx`, `name` (메뉴명), `price` |
| `data/kakao_diner_open_hours.csv` | 688,428 | `diner_idx`, `day_of_week`, `is_open`, `start_time`, `end_time` |
| `data/ai_df.csv` | 67,929 | `diner_idx`, `ai_bottom_sheet_summary`, `blog_summaries` |

### 목표 로그 스키마 (가상 생성 대상)

```
event_type, event_timestamp, user_id,  session_id,  diner_idx,  search_query,    position
impression, 1767837010973005, 93061099, 1767836225,  diner_A,   매봉역 맛집,       1
click,       1767837010973100, 93061099, 1767836225,  diner_B,   매봉역 맛집,       5
impression, 1767837010973200, 93061099, 1767836225,  diner_C,   매봉역 맛집,       12
```

---

## 핵심 아이디어: 리뷰 → 가상 검색 로그 변환

실제 검색 로그 없이도 **리뷰 데이터가 암묵적인 클릭 신호**를 담고 있다.

> "유저가 리뷰를 작성했다 = 그 식당을 방문했다 = 어떤 검색 의도가 있었을 것이다"

### 변환 로직

```
1. review.csv의 (reviewer_id, diner_idx) 쌍 → 클릭(Positive)
2. 같은 지역 + 카테고리의 미방문 식당 → 노출(Negative)
3. diner_road_address에서 지역 키워드 추출 → search_query 생성
4. bayesian_score 내림차순 정렬 → position 할당
```

### 쿼리 생성 규칙

`diner_road_address`와 `diner_category_large`를 조합해 대표 쿼리 패턴을 생성한다.

```python
# 주소 예시: "서울특별시 강남구 선릉로86길 30"
# → 구(gu)  : "강남"
# → 동(dong): None (없는 경우)

# 쿼리 후보 템플릿
QUERY_TEMPLATES = [
    "{구}",                        # "강남"
    "{구} 맛집",                   # "강남 맛집"
    "{구} {카테고리_대분류}",        # "강남 한식"
    "{구} {카테고리_중분류}",        # "강남 삼겹살"
    "{구} {카테고리_대분류} 맛집",   # "강남 한식 맛집"
    "{동}",                        # "역삼"  (동 정보 있을 경우)
    "{동} 맛집",                   # "역삼 맛집"
    "{동} {카테고리_중분류}",        # "역삼 삼겹살"
]
```

각 리뷰에 대해 하나의 쿼리를 확률적으로 선택한다 (쿼리 다양성 확보).

### 세션 구성

```python
# 같은 날 같은 유저의 리뷰들 → 같은 세션
# session_id = reviewer_id * 10000 + 날짜 해시

# 세션 내 impression 구성:
# - 클릭(방문)한 식당: 해당 지역+카테고리의 bayesian_score 기준 상위 50위 내에서 실제 위치 부여
# - 클릭하지 않은 식당: 동일 지역+카테고리에서 미방문 식당 N개 샘플링
```

---

## 가상 로그 생성 파이프라인

```
data/review.csv
data/diner.csv          ──► scripts/generate_search_logs.py ──► data/search_logs.parquet
data/diner_category.csv
```

### 생성 스크립트 설계 (`scripts/generate_search_logs.py`)

```python
def generate_search_logs(
    review_df: pd.DataFrame,     # review.csv
    diner_df: pd.DataFrame,      # diner.csv
    category_df: pd.DataFrame,   # diner_category.csv
    num_impressions_per_click: int = 19,  # click 1개당 impression 19개 → CTR 5%
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    리뷰 데이터에서 가상 검색 로그를 생성한다.

    각 리뷰 = 1 click event
    동일 지역+카테고리 미방문 식당 N개 = impression events

    Returns:
        columns: [event_type, event_timestamp, user_id, session_id,
                  shop_id, search_query, position, query_id]
    """
```

**생성 데이터 규모 예상**:
- 클릭: ~1.86M (리뷰 수)
- 노출: ~37.2M (클릭 * 20)
- 고유 쿼리 수: ~수만 개 (지역 × 카테고리 조합)

**학습/검증/테스트 분할**:
- 기존 시간 기반 분할과 동일한 기준 적용
- Train: `reviewer_review_date < 2025-06-01`
- Val: `2025-06-01 ~ 2025-09-01`
- Test: `2025-09-01 ~`

---

## Stage 1: Query-Diner Two-Tower Retrieval

### 목표

검색 쿼리와 식당을 같은 128차원 임베딩 공간에 매핑하여
ANN(Approximate Nearest Neighbor) 검색으로 후보를 빠르게 뽑는다.

### 아키텍처

```
  Query Tower (신규)               Diner Tower (기존 multimodal_triplet 재사용)
  ─────────────────                ─────────────────────────────────────────────
  search_query                     category + menu + diner_name + price + review
       │                                             │
  KoBERT (klue/bert-base, frozen)        각 Encoder (기존과 동일)
       │                                             │
  Mean Pooling → 768-d                    Attention Fusion
       │                                             │
  MLP (768 → 256 → 128)             Final MLP (608 → 256 → 128)
       │                                             │
  L2 Normalize                           L2 Normalize
       │                                             │
  query_emb (128-d)                    diner_emb (128-d)
       └──────────────────┬────────────────┘
                          │
                  dot product similarity
                  (L2 norm → cosine sim)
```

**Diner Tower 전략**:
기존 `MultimodalTripletModel`의 사전학습 가중치를 **freeze**한다.
Query Tower만 학습하여 쿼리 임베딩이 관련 식당 임베딩에 가까워지도록 한다.

### 학습 데이터

가상 로그에서 생성:
```
Positive pair: (search_query, clicked_diner_idx)
Negative: in-batch negatives (동일 배치 내 다른 쿼리의 clicked_diner들)
```

기존 `MultimodalTripletDataset`과 동일한 InfoNCE 방식 적용:
- 배치 내 알려진 positive pair는 `-inf` 마스킹 (false negative 방지)
- 배치 크기 256 권장 (GPU 메모리 고려)

### 학습 설정

| 파라미터 | 값 |
|----------|-----|
| Loss | In-batch InfoNCE (기존 `src/yamyam_lab/loss/infonce.py` 재사용) |
| Temperature | 0.07 |
| Batch size | 256 |
| Query encoder | KoBERT frozen + MLP (학습 대상) |
| Diner encoder | pretrained 가중치 freeze |
| Optimizer | AdamW (lr=5e-4, weight_decay=1e-5) |
| Gradient clipping | 1.0 |
| Early stopping | patience=10 |

### 인퍼런스 (FAISS ANN Search)

```python
# 1. 사전에 모든 식당 임베딩을 FAISS 인덱스에 저장
diner_index = faiss.IndexFlatIP(128)   # Inner Product = cosine (L2 norm 적용 후)
diner_index.add(all_diner_embeddings)  # (num_diners=124111, 128)

# 2. 쿼리 임베딩 생성
query_emb = query_tower.encode("강남 한식 맛집")  # (1, 128)

# 3. Top-K 후보 검색
scores, indices = diner_index.search(query_emb, k=100)
```

---

## Stage 2: Search LightGBM Ranker

### 목표

Stage 1에서 뽑은 Top-K 후보를 최종 검색 결과로 정렬한다.

- 레이블: 클릭(1) / 노출만(0)
- 그룹: `query_id` 단위 LambdaRank
- 기존 `RankerDatasetLoader`, `LightGBMTrainer` 구조를 최대한 재사용

### 피처 설계

#### (A) Query-Diner 관련도 피처 (신규)

| 피처 | 계산 방법 | 비고 |
|------|-----------|------|
| `retrieval_score` | Stage 1 dot product 점수 | 주요 신호 |
| `query_region_in_address` | 쿼리의 구/동 키워드가 `diner_road_address`에 포함 여부 (0/1) | 지역 매칭 |
| `query_category_match` | 쿼리 카테고리 키워드 ↔ `diner_category_large/middle` 일치 여부 | 카테고리 매칭 |
| `query_diner_name_overlap` | 쿼리 토큰 ↔ `diner_name` 형태소 Jaccard | 이름 유사도 |
| `query_menu_overlap` | 쿼리 키워드 ↔ `menu_df.name` 매칭 여부 | 메뉴 매칭 |

#### (B) 식당 피처 (기존 재사용)

기존 `DinerFeatureStore`에서 엔지니어링된 피처를 그대로 활용한다.

| 피처 | 출처 |
|------|------|
| `min_price`, `max_price`, `mean_price`, `median_price` | `menu_df.csv` |
| `open_days_per_week`, `is_open_weekend`, `avg_open_hours_per_day` | `kakao_diner_open_hours.csv` |
| `korean_ratio`, `japanese_ratio`, `chinese_ratio`, `western_ratio`, `asian_ratio` | `diner_category.csv` |
| `log_total_visits` | `review.csv` 집계 |
| `bayesian_score` | `diner.csv` |
| `taste`, `kind`, `mood`, `chip`, `parking` | `review.csv` 태그 집계 |

#### (C) Position Bias 피처 (신규)

| 피처 | 설명 |
|------|------|
| `log_position` | `log(position + 1)`, 가상 로그에서의 노출 위치 |
| `shop_impression_cnt` | 전체 노출 횟수 |

> **주의**: 가상 로그에서 position은 `bayesian_score` 기준 생성이므로 실제 편향 정도가 제한적이다.
> 실제 로그가 확보되면 IPS(Inverse Propensity Score) 보정을 적용한다.

### 레이블 구성 로직

```python
# query_id = session_id + "_" + search_query.strip().lower()
logs["query_id"] = (
    logs["session_id"].astype(str) + "_" + logs["search_query"].str.strip().str.lower()
)

# 동일 query_id 내에서 click 유무로 레이블 결정
clicks = logs[logs["event_type"] == "click"][["query_id", "shop_id"]].assign(label=1)
impressions = logs[logs["event_type"] == "impression"][["query_id", "shop_id", "position"]]

labeled = impressions.merge(clicks, on=["query_id", "shop_id"], how="left")
labeled["label"] = labeled["label"].fillna(0).astype(int)
```

### 기존 코드 재사용 전략

```
기존 RankerDatasetLoader
  group_column: reviewer_id  →  query_id 로 교체
  candidate_type: node2vec   →  two_tower_search 로 추가
  target_column: target      →  label (click 여부)
```

```
기존 LightGBMTrainer
  objective: lambdarank (유지)
  metric: ndcg (유지)
  group: query_id 기준으로 _get_groups() 동작 (reviewer_id → query_id)
```

---

## 구현 로드맵

### Phase 1: 가상 로그 생성

- [ ] `scripts/generate_search_logs.py`
  - 주소에서 구/동 키워드 추출 (정규식 + 행정구역 사전)
  - 쿼리 템플릿 기반 search_query 생성
  - 세션 단위 impression/click 이벤트 구성
  - `data/search_logs.parquet` 저장

### Phase 2: Stage 1 — Query Tower

- [ ] `src/yamyam_lab/model/embedding/query_tower.py`
  - KoBERT(frozen) + MLP(768→256→128) + L2 norm
- [ ] `src/yamyam_lab/model/embedding/two_tower_search.py`
  - Query Tower + 기존 Diner Tower 연결
  - InfoNCE loss (기존 `src/yamyam_lab/loss/infonce.py` 재사용)
- [ ] `src/yamyam_lab/data/two_tower_search.py`
  - `(query, clicked_diner_idx)` positive pair Dataset
- [ ] `src/yamyam_lab/engine/two_tower_search_trainer.py`
  - 기존 `MultimodalTripletTrainer` 패턴 참고
- [ ] `config/models/embedding/two_tower_search.yaml`
- [ ] `scripts/build_search_faiss_index.py`
  - 학습된 Diner Tower로 124K 식당 임베딩 추출 → FAISS 인덱스 저장

### Phase 3: Stage 2 — Search Ranker

- [ ] `src/yamyam_lab/data/search_ranker.py`
  - `SearchRankerDatasetLoader`: 가상 로그 기반 학습 데이터 구성
  - `group_column=query_id`, Stage 1 retrieval_score 머지
- [ ] `src/yamyam_lab/features/search_query.py`
  - query-region, query-category, query-name, query-menu 관련도 피처
- [ ] `config/models/ranker/search_lightgbm.yaml`
  - 기존 `lightgbm.yaml` 확장, `group_column: query_id` 추가
- [ ] `src/yamyam_lab/search_rerank.py`
  - 기존 `rerank.py` 패턴 유지, 검색용 데이터 로더로 교체

### Phase 4: 평가

- [ ] Recall@K: Stage 1 — Top-K 후보 내 실제 클릭 식당 포함률
- [ ] NDCG@K: Stage 2 — 최종 랭킹 품질 (K=5, 10, 20)
- [ ] `docs/architecture/evaluation.md` 업데이트

---

## 핵심 설계 결정사항

### 1. Diner Tower Freeze 전략

| 옵션 | 장점 | 단점 |
|------|------|------|
| **A. Full freeze (권장)** | 빠른 학습, 기존 추천 성능 보존 | 검색 특화 표현 학습 불가 |
| B. Fine-tune (낮은 lr) | 검색 도메인 적응 | 기존 유사 식당 임베딩 성능 저하 위험 |
| C. 별도 Diner Tower 학습 | 태스크 특화 | 파라미터 2배, 관리 복잡 |

초기에는 A로 시작하여 Recall@K 성능 확인 후 B 검토.

### 2. 가상 로그의 한계 및 보완

| 한계 | 보완 방법 |
|------|-----------|
| 실제 쿼리 분포와 다를 수 있음 | 쿼리 템플릿 다양화, 실제 로그 확보 시 교체 |
| Position은 bayesian_score 기반으로 실제 편향과 다름 | `log_position` 피처로 모델이 학습하도록 유도 |
| 동일 세션에 여러 카테고리 리뷰 → 쿼리 불일치 | 세션 내 단일 지역+카테고리로 제한 |

### 3. 쿼리 정규화

"양재"와 "양재역", "양재/도곡 맛집"은 의미적으로 유사하지만 문자열이 다름.
가상 로그 생성 단계에서 템플릿 기반으로 쿼리를 생성하므로 어느 정도 정규화됨.
실제 로그 적용 시 Kiwi 형태소 분석으로 NNP(지역명) + NNG(키워드) 추출 필요.

---

## 파일 참조

| 파일 | 상태 | 설명 |
|------|------|------|
| `src/yamyam_lab/model/embedding/multimodal_triplet.py` | 기존 재사용 | Diner Tower |
| `src/yamyam_lab/loss/infonce.py` | 기존 재사용 | InfoNCE loss + positive mask |
| `src/yamyam_lab/model/embedding/query_tower.py` | **신규** | Query Tower |
| `src/yamyam_lab/model/embedding/two_tower_search.py` | **신규** | Two-Tower 통합 모델 |
| `src/yamyam_lab/data/two_tower_search.py` | **신규** | (query, diner) pair Dataset |
| `src/yamyam_lab/engine/two_tower_search_trainer.py` | **신규** | Two-Tower Trainer |
| `src/yamyam_lab/data/search_ranker.py` | **신규** | 검색 랭커용 데이터 로더 |
| `src/yamyam_lab/features/search_query.py` | **신규** | 쿼리-식당 관련도 피처 |
| `src/yamyam_lab/search_rerank.py` | **신규** | 검색 reranking 엔트리포인트 |
| `config/models/embedding/two_tower_search.yaml` | **신규** | Two-Tower 설정 |
| `config/models/ranker/search_lightgbm.yaml` | **신규** | 검색 LightGBM 설정 |
| `scripts/generate_search_logs.py` | **신규** | 가상 로그 생성 스크립트 |
| `scripts/build_search_faiss_index.py` | **신규** | FAISS 인덱스 빌드 |
