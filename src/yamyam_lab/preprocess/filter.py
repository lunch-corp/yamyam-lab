import logging
import time
from collections import Counter
from typing import List

import pandas as pd
from kiwipiepy import Kiwi
from tools.morpheme import tokenize_with_kiwi


class Filter:
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize Filter class when preprocessing step.
        """
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.kiwi = Kiwi(typos="basic_with_continual_and_lengthening")

    def filter_martial_law_reviews(
        self,
        review: pd.DataFrame,
        target_months: List[str],
        min_common_word_count_with_abusive_words: int,
        min_review_count_by_diner_id: int,
        included_tags: List[str],
        abusive_words: List[str],
        pre_calculated_diner_ids: List[int],
    ) -> pd.DataFrame:
        """
        Filter abusive reviews talking about martial laws when 2024-12 ~ 2025-01.

        Args:
            review (pd.DataFrame): Raw review data before filtering.
            target_months (List[str]): Target months when martial law happened.
            min_common_word_count_with_abusive_words (int): Minimum overlapping word counts with pre-defined abusive words.
            min_review_count_by_diner_id (int): Minimum review count which has abusive words.
            included_tags (List[str]): Morpheme tags to be included.
            abusive_words (List[str]): Pre-defined abusive words related with martial law.
            pre_calculated_diner_ids (List[int]): Pre-calculated diner_ids with abusive reviews to skip calculation.

        Returns (pd.DataFrame):
            Review after excluding abusive reviews with martial law.
        """
        # target date when martial law happened
        review_filtered = review[
            lambda x: x["reviewer_review_date"].map(lambda x: x[:7]).isin(target_months)
        ]
        # for later concatenation
        review_not_in_target_months = review[
            lambda x: ~x["reviewer_review_date"]
            .map(lambda x: x[:7])
            .isin(target_months)
        ]
        if len(pre_calculated_diner_ids) == 0:
            # tokenize review and count them
            start = time.time()
            review_filtered["token_count"] = review_filtered["reviewer_review"].map(
                lambda x: Counter(tokenize_with_kiwi(self.kiwi, x, included_tags))
            )
            # Note: this code line takes about 220 seconds.
            self.logger.info(
                f"Token time for tokenizing: {round(time.time() - start, 2)}"
            )

            # tag if each review is talking about martial law or not by overlapping word
            review_filtered["is_martial_law_review"] = review_filtered[
                "token_count"
            ].map(lambda x: len(set(x.keys()) & set(abusive_words)) >= 1)
            # sum martial law review counts by diner_id
            martial_law_review_count_by_diner_id = (
                review_filtered.groupby("diner_idx")["is_martial_law_review"]
                .sum()
                .to_dict()
            )

            # sum token counts by diner_id
            token_count_by_diner_id = (
                review_filtered.groupby("diner_idx")["token_count"].sum().to_dict()
            )

            # apply abusing detection rule
            # [Rule]
            # 1. after aggregation token count by diner_id, if there are more or equal min_common_word_count_with_abusive_words words
            # 2. if there are more or equal min_review_count_by_diner_id reviews containing martial law tokens
            diner_ids_with_martial_law_reviews = []
            for diner_id, token_count in token_count_by_diner_id.items():
                if (
                    len(set(token_count.keys()) & set(abusive_words))
                    >= min_common_word_count_with_abusive_words
                    and martial_law_review_count_by_diner_id.get(diner_id, 0)
                    >= min_review_count_by_diner_id
                ):
                    diner_ids_with_martial_law_reviews.append(diner_id)
        else:
            diner_ids_with_martial_law_reviews = pre_calculated_diner_ids[:]
            self.logger.info("Using pre-calculated diner_ids with abusive reviews")
        self.logger.info(
            f"Detected {len(diner_ids_with_martial_law_reviews)} diner_ids with abusive reviews: {diner_ids_with_martial_law_reviews}"
        )

        # exclude and concatenate to original data
        preprocessed_review = pd.concat(
            [
                review_not_in_target_months,
                review_filtered[
                    lambda x: ~x["diner_idx"].isin(diner_ids_with_martial_law_reviews)
                ],
            ]
        )
        self.logger.info(
            f"Excluded {review.shape[0] - preprocessed_review.shape[0]} abusive reviews"
        )
        return preprocessed_review
