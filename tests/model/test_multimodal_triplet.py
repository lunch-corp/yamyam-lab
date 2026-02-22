"""Tests for multimodal triplet embedding model training pipeline."""

import argparse
import os
from unittest.mock import patch

import matplotlib
import pytest
import torch
import torch.nn.functional as F
from easydict import EasyDict

from yamyam_lab.data.multimodal_triplet import (
    MultimodalTripletDataset,
    create_multimodal_triplet_dataloader,
    multimodal_triplet_collate_fn,
)
from yamyam_lab.engine.multimodal_triplet_trainer import MultimodalTripletTrainer
from yamyam_lab.loss.triplet import (
    batch_hard_triplet_loss,
    triplet_margin_loss,
    triplet_margin_loss_with_category,
    triplet_margin_loss_with_multiple_negatives,
)
from yamyam_lab.model.embedding.encoders import (
    AttentionFusion,
    CategoryEncoder,
    DinerNameEncoder,
    FinalProjection,
    MenuEncoder,
    PriceEncoder,
)
from yamyam_lab.model.embedding.multimodal_triplet import Model
from yamyam_lab.train import TrainerFactory

matplotlib.use("Agg")


class TestEncoders:
    """Unit tests for encoder submodules."""

    def test_category_encoder_output_shape(self):
        """CategoryEncoder produces (batch_size, category_dim) output."""
        encoder = CategoryEncoder(
            num_large_categories=3,
            num_middle_categories=5,
            output_dim=32,
        )
        out = encoder(
            large_category_ids=torch.tensor([0, 1, 2, 0]),
            middle_category_ids=torch.tensor([0, 1, 2, 3]),
        )
        assert out.shape == (4, 32)

    def test_menu_encoder_forward_precomputed_shape(self):
        """MenuEncoder.forward_precomputed produces (batch_size, menu_dim) output."""
        encoder = MenuEncoder(output_dim=64, dropout=0.0)
        out = encoder.forward_precomputed(torch.randn(4, 768))
        assert out.shape == (4, 64)

    def test_diner_name_encoder_forward_precomputed_shape(self):
        """DinerNameEncoder.forward_precomputed produces (batch_size, diner_name_dim)."""
        encoder = DinerNameEncoder(output_dim=16, dropout=0.0)
        out = encoder.forward_precomputed(torch.randn(4, 768))
        assert out.shape == (4, 16)

    def test_price_encoder_output_shape(self):
        """PriceEncoder produces (batch_size, price_dim) output."""
        encoder = PriceEncoder(output_dim=8)
        out = encoder(torch.randn(4, 3))
        assert out.shape == (4, 8)

    def test_attention_fusion_output_shape(self):
        """AttentionFusion produces (batch_size, total_dim) output."""
        fusion = AttentionFusion(
            category_dim=32, menu_dim=64, diner_name_dim=16, price_dim=8, num_heads=2
        )
        out = fusion(
            category_emb=torch.randn(4, 32),
            menu_emb=torch.randn(4, 64),
            diner_name_emb=torch.randn(4, 16),
            price_emb=torch.randn(4, 8),
        )
        assert out.shape == (4, 32 + 64 + 16 + 8)

    def test_final_projection_l2_normalized(self):
        """FinalProjection output is L2-normalized (norm ~= 1.0 per row)."""
        proj = FinalProjection(input_dim=120, output_dim=32, dropout=0.0)
        out = proj(torch.randn(4, 120))
        norms = torch.norm(out, p=2, dim=-1)
        assert out.shape == (4, 32)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)


class TestModel:
    """Unit tests for Model class."""

    def test_forward_output_shape(self, small_model_config):
        """Model.forward produces (batch_size, embedding_dim) output."""
        model = Model(small_model_config)
        features = {
            "large_category_ids": torch.tensor([0, 1, 2, 0]),
            "middle_category_ids": torch.tensor([0, 1, 2, 3]),
            "menu_embeddings": torch.randn(4, 768),
            "diner_name_embeddings": torch.randn(4, 768),
            "price_features": torch.randn(4, 3),
            "review_text_embeddings": torch.randn(4, 768),
        }
        out = model(features)
        assert out.shape == (4, 32)

    def test_forward_output_is_l2_normalized(self, small_model_config):
        """Model output embeddings have unit L2 norm."""
        model = Model(small_model_config)
        features = {
            "large_category_ids": torch.tensor([0, 1]),
            "middle_category_ids": torch.tensor([0, 1]),
            "menu_embeddings": torch.randn(2, 768),
            "diner_name_embeddings": torch.randn(2, 768),
            "price_features": torch.randn(2, 3),
            "review_text_embeddings": torch.randn(2, 768),
        }
        out = model(features)
        norms = torch.norm(out, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_compute_and_store_embeddings(self, small_model_config):
        """compute_and_store_embeddings populates _embedding with correct shape."""
        model = Model(small_model_config)
        n = 10
        all_features = {
            "large_category_ids": torch.randint(0, 3, (n,)),
            "middle_category_ids": torch.randint(0, 5, (n,)),
            "menu_embeddings": torch.randn(n, 768),
            "diner_name_embeddings": torch.randn(n, 768),
            "price_features": torch.randn(n, 3),
            "review_text_embeddings": torch.randn(n, 768),
        }
        model.compute_and_store_embeddings(all_features, batch_size=4)
        assert model._embedding is not None
        assert model._embedding.shape == (n, 32)

    def test_get_embedding_raises_without_compute(self, small_model_config):
        """get_embedding raises RuntimeError if embeddings not computed."""
        model = Model(small_model_config)
        with pytest.raises(RuntimeError, match="Embeddings not computed"):
            model.get_embedding(torch.tensor([0]))

    def test_similarity_shape(self, small_model_config):
        """similarity returns (batch_size, num_candidates) shape."""
        model = Model(small_model_config)
        sim = model.similarity(torch.randn(3, 32), torch.randn(10, 32))
        assert sim.shape == (3, 10)

    def test_recommend_returns_correct_format(self, small_model_config):
        """recommend returns (indices, scores) with correct shape."""
        model = Model(small_model_config)
        n = 10
        all_features = {
            "large_category_ids": torch.randint(0, 3, (n,)),
            "middle_category_ids": torch.randint(0, 5, (n,)),
            "menu_embeddings": torch.randn(n, 768),
            "diner_name_embeddings": torch.randn(n, 768),
            "price_features": torch.randn(n, 3),
            "review_text_embeddings": torch.randn(n, 768),
        }
        model.compute_and_store_embeddings(all_features, batch_size=4)
        indices, scores = model.recommend(
            model._embedding[0:1], exclude_indices=[0], top_k=3
        )
        assert len(indices) == 3
        assert len(scores) == 3
        assert 0 not in indices

    def test_recommend_raises_without_compute(self, small_model_config):
        """recommend raises RuntimeError if embeddings not computed."""
        model = Model(small_model_config)
        with pytest.raises(RuntimeError, match="Embeddings not computed"):
            model.recommend(torch.randn(1, 32))

    def test_generate_candidates_for_each_diner(self, small_model_config):
        """generate_candidates returns DataFrame with correct columns."""
        model = Model(small_model_config)
        n = 10
        all_features = {
            "large_category_ids": torch.randint(0, 3, (n,)),
            "middle_category_ids": torch.randint(0, 5, (n,)),
            "menu_embeddings": torch.randn(n, 768),
            "diner_name_embeddings": torch.randn(n, 768),
            "price_features": torch.randn(n, 3),
            "review_text_embeddings": torch.randn(n, 768),
        }
        model.compute_and_store_embeddings(all_features, batch_size=4)
        df = model.generate_candidates_for_each_diner(top_k_value=3)
        assert set(df.columns) == {"diner_id", "candidate_diner_id", "score"}
        assert len(df) == n * 3
        assert (df["diner_id"] != df["candidate_diner_id"]).all()


class TestMultimodalTripletDataset:
    """Unit tests for MultimodalTripletDataset."""

    def test_dataset_length(self, multimodal_triplet_parquet_data):
        """Dataset length equals number of training pairs."""
        paths = multimodal_triplet_parquet_data
        dataset = MultimodalTripletDataset(
            features_path=paths["features_path"],
            pairs_path=paths["pairs_path"],
            category_mapping_path=paths["category_mapping_path"],
        )
        assert len(dataset) == 30

    def test_getitem_returns_correct_keys(self, multimodal_triplet_parquet_data):
        """__getitem__ returns dict with all expected keys."""
        paths = multimodal_triplet_parquet_data
        dataset = MultimodalTripletDataset(
            features_path=paths["features_path"],
            pairs_path=paths["pairs_path"],
            category_mapping_path=paths["category_mapping_path"],
        )
        sample = dataset[0]
        expected_keys = {
            "anchor_idx",
            "positive_idx",
            "anchor_diner_idx",
            "positive_diner_idx",
        }
        assert set(sample.keys()) == expected_keys

    def test_getitem_indices_are_scalars(self, multimodal_triplet_parquet_data):
        """All returned tensors are scalars (0-dim)."""
        paths = multimodal_triplet_parquet_data
        dataset = MultimodalTripletDataset(
            features_path=paths["features_path"],
            pairs_path=paths["pairs_path"],
            category_mapping_path=paths["category_mapping_path"],
        )
        sample = dataset[0]
        for key in sample:
            assert sample[key].dim() == 0

    def test_get_all_features_shape(self, multimodal_triplet_parquet_data):
        """get_all_features returns tensors with correct shapes."""
        paths = multimodal_triplet_parquet_data
        dataset = MultimodalTripletDataset(
            features_path=paths["features_path"],
            pairs_path=paths["pairs_path"],
            category_mapping_path=paths["category_mapping_path"],
        )
        features = dataset.get_all_features()
        n = dataset.num_diners
        assert features["large_category_ids"].shape == (n,)
        assert features["menu_embeddings"].shape == (n, 768)
        assert features["diner_name_embeddings"].shape == (n, 768)
        assert features["price_features"].shape == (n, 3)
        assert features["review_text_embeddings"].shape == (n, 768)

    def test_get_features_by_indices(self, multimodal_triplet_parquet_data):
        """get_features_by_indices returns correct subset."""
        paths = multimodal_triplet_parquet_data
        dataset = MultimodalTripletDataset(
            features_path=paths["features_path"],
            pairs_path=paths["pairs_path"],
            category_mapping_path=paths["category_mapping_path"],
        )
        indices = torch.tensor([0, 2, 4])
        features = dataset.get_features_by_indices(indices)
        assert features["large_category_ids"].shape == (3,)
        assert features["menu_embeddings"].shape == (3, 768)

    def test_diner_to_positives_populated(self, multimodal_triplet_parquet_data):
        """diner_to_positives is populated after loading."""
        paths = multimodal_triplet_parquet_data
        dataset = MultimodalTripletDataset(
            features_path=paths["features_path"],
            pairs_path=paths["pairs_path"],
            category_mapping_path=paths["category_mapping_path"],
        )
        assert len(dataset.diner_to_positives) > 0
        # Each value should be a set
        for v in dataset.diner_to_positives.values():
            assert isinstance(v, set)


class TestCollateFunction:
    """Tests for collate function and DataLoader creation."""

    def test_collate_fn_batches_correctly(self, multimodal_triplet_parquet_data):
        """Collate function produces correctly batched tensors."""
        paths = multimodal_triplet_parquet_data
        dataset = MultimodalTripletDataset(
            features_path=paths["features_path"],
            pairs_path=paths["pairs_path"],
            category_mapping_path=paths["category_mapping_path"],
        )
        batch = [dataset[i] for i in range(4)]
        collated = multimodal_triplet_collate_fn(batch)
        assert collated["anchor_indices"].shape == (4,)
        assert collated["positive_indices"].shape == (4,)
        assert collated["anchor_diner_indices"].shape == (4,)
        assert collated["positive_diner_indices"].shape == (4,)

    def test_dataloader_iterates(self, multimodal_triplet_parquet_data):
        """DataLoader from factory function yields valid batches."""
        paths = multimodal_triplet_parquet_data
        dl, ds = create_multimodal_triplet_dataloader(
            features_path=paths["features_path"],
            pairs_path=paths["pairs_path"],
            category_mapping_path=paths["category_mapping_path"],
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(dl))
        assert "anchor_indices" in batch
        assert "anchor_diner_indices" in batch
        assert batch["anchor_indices"].shape[0] <= 4


class TestTripletLoss:
    """Unit tests for triplet loss functions."""

    def test_triplet_margin_loss_non_negative(self):
        """Loss is always non-negative."""
        anchor = F.normalize(torch.randn(8, 32), p=2, dim=-1)
        positive = F.normalize(torch.randn(8, 32), p=2, dim=-1)
        negative = F.normalize(torch.randn(8, 32), p=2, dim=-1)
        loss = triplet_margin_loss(anchor, positive, negative, margin=0.5)
        assert loss.item() >= 0.0

    def test_triplet_margin_loss_positive_when_negative_closer(self):
        """Loss is positive when negative is closer than positive."""
        anchor = F.normalize(torch.tensor([[1.0, 0.0]]), p=2, dim=-1)
        positive = F.normalize(torch.tensor([[-1.0, 0.0]]), p=2, dim=-1)
        negative = F.normalize(torch.tensor([[0.9, 0.1]]), p=2, dim=-1)
        loss = triplet_margin_loss(anchor, positive, negative, margin=0.5)
        assert loss.item() > 0.0

    def test_triplet_margin_loss_is_differentiable(self):
        """Loss supports gradient computation."""
        anchor = F.normalize(torch.randn(4, 32, requires_grad=True), p=2, dim=-1)
        positive = F.normalize(torch.randn(4, 32), p=2, dim=-1)
        negative = F.normalize(torch.randn(4, 32), p=2, dim=-1)
        loss = triplet_margin_loss(anchor, positive, negative)
        loss.backward()

    def test_triplet_margin_loss_with_category_returns_scalar(self):
        """Category-aware loss returns a scalar tensor."""
        anchor = F.normalize(torch.randn(4, 32), p=2, dim=-1)
        positive = F.normalize(torch.randn(4, 32), p=2, dim=-1)
        negative = F.normalize(torch.randn(4, 32), p=2, dim=-1)
        cats = torch.tensor([0, 0, 1, 1])
        loss = triplet_margin_loss_with_category(
            anchor, positive, negative, cats, cats, cats, margin=0.5
        )
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_triplet_margin_loss_with_multiple_negatives_returns_scalar(self):
        """Multiple-negatives loss returns a scalar."""
        B, D, N = 4, 32, 3
        anchor = F.normalize(torch.randn(B, D), p=2, dim=-1)
        positive = F.normalize(torch.randn(B, D), p=2, dim=-1)
        negatives = F.normalize(torch.randn(B, N, D), p=2, dim=-1)
        a_cat = torch.tensor([0, 0, 1, 1])
        p_cat = torch.tensor([0, 0, 1, 1])
        n_cat = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
        loss = triplet_margin_loss_with_multiple_negatives(
            anchor, positive, negatives, a_cat, p_cat, n_cat, margin=0.5
        )
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_batch_hard_triplet_loss_zero_with_same_labels(self):
        """batch_hard_triplet_loss returns 0 when all labels are the same."""
        emb = F.normalize(torch.randn(4, 32), p=2, dim=-1)
        labels = torch.tensor([0, 0, 0, 0])
        loss = batch_hard_triplet_loss(emb, labels, margin=0.5)
        assert loss.item() == 0.0

    def test_batch_hard_triplet_loss_with_valid_triplets(self):
        """batch_hard_triplet_loss returns non-negative value with mixed labels."""
        emb = F.normalize(torch.randn(8, 32), p=2, dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = batch_hard_triplet_loss(emb, labels, margin=0.5)
        assert loss.item() >= 0.0


class TestMultimodalTripletTrainer:
    """Integration tests for the full training pipeline."""

    def test_factory_creates_correct_trainer(self):
        """TrainerFactory creates MultimodalTripletTrainer for multimodal_triplet."""
        args = argparse.Namespace(model="multimodal_triplet")
        trainer = TrainerFactory.create_trainer(args)
        assert isinstance(trainer, MultimodalTripletTrainer)

    def test_full_training_pipeline(
        self, tmp_path, multimodal_triplet_parquet_data, multimodal_triplet_config
    ):
        """Integration test: full train() pipeline completes without error."""
        result_path = str(tmp_path / "results")
        os.makedirs(result_path, exist_ok=True)

        args = argparse.Namespace(
            model="multimodal_triplet",
            device="cpu",
            num_workers=0,
            random_seed=42,
            epochs=2,
            lr=0.001,
            batch_size=8,
            patience=2,
            result_path=result_path,
            config_root_path=None,
            postfix="pytest",
            test=True,
            save_candidate=False,
        )

        config = multimodal_triplet_config

        with (
            patch(
                "yamyam_lab.engine.base_trainer.load_configs",
                return_value=(config, EasyDict({})),
            ),
            patch(
                "yamyam_lab.engine.base_trainer.generate_result_path",
                return_value=result_path,
            ),
            patch(
                "yamyam_lab.tools.parse_args.save_command_to_file",
            ),
        ):
            trainer = TrainerFactory.create_trainer(args)
            trainer.train()

        assert os.path.exists(os.path.join(result_path, "log.log"))
        assert len(trainer.model.tr_loss) == 2

    def test_training_without_validation(
        self, tmp_path, multimodal_triplet_parquet_data, multimodal_triplet_config
    ):
        """Training works when val_pairs_path doesn't exist."""
        result_path = str(tmp_path / "results_noval")
        os.makedirs(result_path, exist_ok=True)

        config = multimodal_triplet_config
        config.data.val_pairs_path = str(tmp_path / "nonexistent_val_pairs.parquet")

        args = argparse.Namespace(
            model="multimodal_triplet",
            device="cpu",
            num_workers=0,
            random_seed=42,
            epochs=1,
            lr=0.001,
            batch_size=8,
            patience=1,
            result_path=result_path,
            config_root_path=None,
            postfix="pytest",
            test=True,
            save_candidate=False,
        )

        with (
            patch(
                "yamyam_lab.engine.base_trainer.load_configs",
                return_value=(config, EasyDict({})),
            ),
            patch(
                "yamyam_lab.engine.base_trainer.generate_result_path",
                return_value=result_path,
            ),
            patch(
                "yamyam_lab.tools.parse_args.save_command_to_file",
            ),
        ):
            trainer = TrainerFactory.create_trainer(args)
            trainer.train()

        assert trainer.val_loader is None
