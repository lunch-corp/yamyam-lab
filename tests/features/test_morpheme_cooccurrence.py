"""Tests for morpheme extraction and Jaccard pair generation."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from yamyam_lab.features.morpheme_cooccurrence import (
    build_morpheme_matrix,
    compute_jaccard_pairs,
)


class TestBuildMorphemeMatrix:
    """Tests for build_morpheme_matrix."""

    @pytest.fixture
    def diner_morphemes(self):
        return {
            0: {"맛있다", "친절하다", "깔끔하다"},
            1: {"맛있다", "양많다", "저렴하다"},
            2: {"맛있다", "친절하다", "고급스럽다"},
            3: {"양많다", "저렴하다", "넓다"},
            4: {"깔끔하다", "고급스럽다", "분위기좋다"},
        }

    def test_output_shapes(self, diner_morphemes):
        """Matrix shape is (num_diners, vocab_size)."""
        mat, vocab, idx_to_pos = build_morpheme_matrix(
            diner_morphemes, min_df=1, max_df=1.0
        )
        assert mat.shape[0] == 5
        assert mat.shape[1] == len(vocab)
        assert len(idx_to_pos) == 5

    def test_binary_values(self, diner_morphemes):
        """Matrix contains only 0 and 1."""
        mat, _, _ = build_morpheme_matrix(diner_morphemes, min_df=1, max_df=1.0)
        values = mat.toarray()
        assert set(np.unique(values)).issubset({0.0, 1.0})

    def test_min_df_filtering(self, diner_morphemes):
        """Morphemes appearing in fewer than min_df diners are excluded."""
        mat_no_filter, vocab_no, _ = build_morpheme_matrix(
            diner_morphemes, min_df=1, max_df=1.0
        )
        mat_filtered, vocab_f, _ = build_morpheme_matrix(
            diner_morphemes, min_df=3, max_df=1.0
        )
        assert len(vocab_f) < len(vocab_no)
        # "맛있다" appears in 3 diners, should survive min_df=3
        assert "맛있다" in vocab_f

    def test_max_df_filtering(self, diner_morphemes):
        """Morphemes appearing in more than max_df fraction are excluded."""
        # "맛있다" appears in 3/5 = 0.6, should be excluded at max_df=0.5
        mat, vocab, _ = build_morpheme_matrix(diner_morphemes, min_df=1, max_df=0.5)
        assert "맛있다" not in vocab

    def test_diner_idx_to_pos_mapping(self, diner_morphemes):
        """diner_idx_to_pos maps each diner to a unique row."""
        _, _, idx_to_pos = build_morpheme_matrix(diner_morphemes, min_df=1, max_df=1.0)
        assert set(idx_to_pos.keys()) == {0, 1, 2, 3, 4}
        assert len(set(idx_to_pos.values())) == 5

    def test_save_and_load(self, diner_morphemes, tmp_path):
        """Matrix and vocab can be saved and reloaded."""
        build_morpheme_matrix(
            diner_morphemes, min_df=1, max_df=1.0, save_dir=str(tmp_path)
        )
        assert (tmp_path / "morpheme_matrix.npz").exists()
        assert (tmp_path / "morpheme_vocab.pkl").exists()

        loaded = sp.load_npz(tmp_path / "morpheme_matrix.npz")
        assert loaded.shape[0] == 5


class TestComputeJaccardPairs:
    """Tests for compute_jaccard_pairs."""

    @pytest.fixture
    def setup_data(self):
        """Create a small morpheme matrix and category DataFrame."""
        # 6 diners, 4 morphemes
        # Diner 0 and 1 share all morphemes (same category) -> jaccard=1.0
        # Diner 2 and 3 share some (same category)
        # Diner 4 and 5 are in different category from 0-3
        data = np.array(
            [
                [1, 1, 1, 0],  # diner 0, cat A
                [1, 1, 1, 0],  # diner 1, cat A
                [1, 0, 0, 1],  # diner 2, cat A
                [0, 0, 1, 1],  # diner 3, cat A
                [1, 1, 0, 0],  # diner 4, cat B
                [0, 0, 1, 1],  # diner 5, cat B
            ],
            dtype=np.float32,
        )
        morpheme_matrix = sp.csr_matrix(data)

        category_df = pd.DataFrame(
            {
                "diner_idx": [0, 1, 2, 3, 4, 5],
                "middle_category_id": [10, 10, 10, 10, 20, 20],
            }
        )

        diner_idx_to_pos = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

        return morpheme_matrix, category_df, diner_idx_to_pos

    def test_same_category_constraint(self, setup_data):
        """Pairs are only generated within the same middle category."""
        mat, cat_df, idx_to_pos = setup_data
        pairs = compute_jaccard_pairs(mat, cat_df, idx_to_pos, threshold=0.0)

        cat_map = dict(zip(cat_df["diner_idx"], cat_df["middle_category_id"]))
        for _, row in pairs.iterrows():
            assert cat_map[row["anchor_idx"]] == cat_map[row["positive_idx"]]

    def test_threshold_filtering(self, setup_data):
        """Higher threshold produces fewer pairs."""
        mat, cat_df, idx_to_pos = setup_data
        pairs_low = compute_jaccard_pairs(mat, cat_df, idx_to_pos, threshold=0.1)
        pairs_high = compute_jaccard_pairs(mat, cat_df, idx_to_pos, threshold=0.8)
        assert len(pairs_high) <= len(pairs_low)

    def test_perfect_overlap_has_jaccard_one(self, setup_data):
        """Diners 0 and 1 (identical morphemes) should have jaccard=1.0."""
        mat, cat_df, idx_to_pos = setup_data
        pairs = compute_jaccard_pairs(mat, cat_df, idx_to_pos, threshold=0.9)
        perfect_pairs = pairs[
            ((pairs["anchor_idx"] == 0) & (pairs["positive_idx"] == 1))
            | ((pairs["anchor_idx"] == 1) & (pairs["positive_idx"] == 0))
        ]
        assert len(perfect_pairs) == 2
        assert all(perfect_pairs["jaccard_score"] == 1.0)

    def test_bidirectional_pairs(self, setup_data):
        """All pairs appear in both directions (A->B and B->A)."""
        mat, cat_df, idx_to_pos = setup_data
        pairs = compute_jaccard_pairs(mat, cat_df, idx_to_pos, threshold=0.0)
        assert len(pairs) % 2 == 0

        pair_set = set(zip(pairs["anchor_idx"], pairs["positive_idx"]))
        for a, p in pair_set:
            assert (p, a) in pair_set

    def test_no_self_pairs(self, setup_data):
        """No pair has anchor_idx == positive_idx."""
        mat, cat_df, idx_to_pos = setup_data
        pairs = compute_jaccard_pairs(mat, cat_df, idx_to_pos, threshold=0.0)
        assert (pairs["anchor_idx"] != pairs["positive_idx"]).all()

    def test_skips_single_diner_categories(self):
        """Categories with only one diner produce no pairs."""
        data = np.array([[1, 1], [1, 0]], dtype=np.float32)
        mat = sp.csr_matrix(data)
        cat_df = pd.DataFrame({"diner_idx": [0, 1], "middle_category_id": [10, 20]})
        idx_to_pos = {0: 0, 1: 1}

        pairs = compute_jaccard_pairs(mat, cat_df, idx_to_pos, threshold=0.0)
        assert len(pairs) == 0

    def test_empty_morphemes_zero_jaccard(self):
        """Diners with no morphemes have jaccard=0 and produce no pairs."""
        data = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.float32)
        mat = sp.csr_matrix(data)
        cat_df = pd.DataFrame(
            {"diner_idx": [0, 1, 2], "middle_category_id": [10, 10, 10]}
        )
        idx_to_pos = {0: 0, 1: 1, 2: 2}

        pairs = compute_jaccard_pairs(mat, cat_df, idx_to_pos, threshold=0.0)
        # Only pairs involving diner 2 can have nonzero jaccard (with itself excluded)
        # Diners 0 and 1 both have empty sets -> union=0 -> jaccard=0
        zero_pairs = pairs[
            (pairs["anchor_idx"].isin([0, 1])) & (pairs["positive_idx"].isin([0, 1]))
        ]
        assert len(zero_pairs) == 0
