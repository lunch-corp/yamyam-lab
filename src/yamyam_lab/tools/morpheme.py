from typing import List

import pandas as pd
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords


def tokenize_with_kiwi(kiwi: Kiwi, text: str, included_tags: List[str]):
    if pd.isna(text):
        return []
    else:
        tokens = kiwi.tokenize(text, stopwords=Stopwords(), normalize_coda=True)
        return [token.form for token in tokens if token.tag in included_tags]
