from typing import Any, Dict, List

import pandas as pd
from kiwipiepy import Kiwi


class ReviewProcessor:
    """
    리뷰 텍스트 처리를 위한 클래스

    DisenHAN (Disentangled Heterogeneous Graph Attention Network) 모델에서
    리뷰 데이터를 전처리하고 문장 단위로 분할하는 기능을 제공합니다.

    Attributes:
        kiwi: KiwiPiePy 형태소 분석기 인스턴스
        sentence_nodes: 처리된 문장 노드들의 리스트
    """

    def __init__(
        self,
        model_type: str = "knlm",
        typos: str = "basic_with_continual_and_lengthening",
    ) -> None:
        """
        ReviewProcessor 초기화

        Args:
            model_type: Kiwi 모델 타입 (기본값: "knlm")
            typos: 오타 처리 방식 (기본값: "basic_with_continual_and_lengthening")
        """
        self.kiwi = Kiwi(model_type=model_type, typos=typos)
        self.sentence_nodes: List[Dict[str, Any]] = []

    def process_reviews(
        self, review_table: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        리뷰 테이블을 처리하여 문장 노드 리스트를 생성

        Args:
            review_table: 리뷰 데이터 리스트
                각 항목은 다음 키를 포함해야 함:
                - review_id: 리뷰 ID
                - reviewer_id: 리뷰어 ID
                - diner_idx: 식당 ID
                - reviewer_review: 리뷰 텍스트

        Returns:
            문장 노드 리스트. 각 노드는 다음 정보를 포함:
            - review_id: 원본 리뷰 ID
            - reviewer_id: 리뷰어 ID
            - diner_id: 식당 ID
            - sentiment: 분할된 문장
        """
        self.sentence_nodes = []

        for row in review_table:
            review_id = row["review_id"]
            reviewer_id = row["reviewer_id"]
            diner_id = row["diner_idx"]
            text = row["reviewer_review"]

            if not text:
                continue

            sentences = self.split_review_into_sentences(text)

            for sentence in sentences:
                self.sentence_nodes.append(
                    {
                        "review_id": review_id,
                        "reviewer_id": reviewer_id,
                        "diner_id": diner_id,
                        "sentiment": sentence,
                    }
                )

        return self.sentence_nodes

    def process_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        pandas DataFrame을 처리하여 문장 노드 리스트를 생성

        Args:
            df: 리뷰 데이터프레임
                다음 컬럼들을 포함해야 함:
                - review_id, reviewer_id, diner_idx, reviewer_review

        Returns:
            문장 노드 리스트
        """
        review_table = df.to_dict("records")
        return self.process_reviews(review_table)

    def split_review_into_sentences(self, text: str) -> List[str]:
        """
        리뷰 텍스트를 문장 단위로 분할

        Args:
            text: 분할할 리뷰 텍스트

        Returns:
            분할된 문장들의 리스트
        """
        splited_reviews = []
        sents = self.kiwi.split_into_sents(
            text, return_tokens=True, normalize_coda=True
        )

        for sent in sents:
            if "\n" in sent.text:
                # 줄바꿈이 있는 경우 추가 분할
                sent_splited = sent.text.split("\n")
                for sent_part in sent_splited:
                    cleaned_sent = sent_part.strip().replace(".", "")
                    if cleaned_sent:  # 빈 문자열 제외
                        splited_reviews.append(cleaned_sent)
            else:
                cleaned_sent = sent.text.strip().replace(".", "")
                if cleaned_sent:  # 빈 문자열 제외
                    splited_reviews.append(cleaned_sent)

        return splited_reviews

    def get_sentence_nodes(self) -> List[Dict[str, Any]]:
        """
        처리된 문장 노드들을 반환

        Returns:
            문장 노드 리스트
        """
        return self.sentence_nodes

    def get_sentence_count(self) -> int:
        """
        처리된 문장의 총 개수를 반환

        Returns:
            문장 개수
        """
        return len(self.sentence_nodes)

    def get_unique_reviewers(self) -> List[int]:
        """
        처리된 데이터에서 고유한 리뷰어 ID 목록을 반환

        Returns:
            고유한 리뷰어 ID 리스트
        """
        return list(set(node["reviewer_id"] for node in self.sentence_nodes))

    def get_unique_diners(self) -> List[int]:
        """
        처리된 데이터에서 고유한 식당 ID 목록을 반환

        Returns:
            고유한 식당 ID 리스트
        """
        return list(set(node["diner_id"] for node in self.sentence_nodes))

    def filter_by_diner(self, diner_id: int) -> List[Dict[str, Any]]:
        """
        특정 식당의 문장 노드들만 필터링

        Args:
            diner_id: 필터링할 식당 ID

        Returns:
            해당 식당의 문장 노드 리스트
        """
        return [node for node in self.sentence_nodes if node["diner_id"] == diner_id]

    def filter_by_reviewer(self, reviewer_id: int) -> List[Dict[str, Any]]:
        """
        특정 리뷰어의 문장 노드들만 필터링

        Args:
            reviewer_id: 필터링할 리뷰어 ID

        Returns:
            해당 리뷰어의 문장 노드 리스트
        """
        return [
            node for node in self.sentence_nodes if node["reviewer_id"] == reviewer_id
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """
        문장 노드들을 pandas DataFrame으로 변환

        Returns:
            문장 노드 데이터프레임
        """
        return pd.DataFrame(self.sentence_nodes)

    def reset(self) -> None:
        """
        처리된 데이터를 초기화
        """
        self.sentence_nodes = []


# 하위 호환성을 위한 기존 함수들 (deprecated)
def process_reviews(review_table):
    """
    Deprecated: ReviewProcessor 클래스를 사용하세요.
    """
    processor = ReviewProcessor()
    return processor.process_reviews(review_table)


def split_review_into_sentences(text):
    """
    Deprecated: ReviewProcessor 클래스를 사용하세요.
    """
    processor = ReviewProcessor()
    return processor.split_review_into_sentences(text)
