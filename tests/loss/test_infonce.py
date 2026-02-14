"""Tests for InfoNCE loss function."""

import pytest
import torch
import torch.nn.functional as F

from yamyam_lab.loss.infonce import infonce_loss_with_multiple_negatives


class TestInfoNCELoss:
    """Tests for infonce_loss_with_multiple_negatives."""

    @pytest.fixture
    def batch_size(self):
        return 8

    @pytest.fixture
    def embedding_dim(self):
        return 128

    @pytest.fixture
    def num_negatives(self):
        return 10

    @pytest.fixture
    def normalized_embeddings(self, batch_size, embedding_dim, num_negatives):
        """Create L2-normalized anchor, positive, and negative embeddings."""
        torch.manual_seed(42)
        anchor = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=-1)
        positive = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=-1)
        negatives = F.normalize(
            torch.randn(batch_size, num_negatives, embedding_dim), p=2, dim=-1
        )
        return anchor, positive, negatives

    @pytest.fixture
    def categories(self, batch_size, num_negatives):
        """Create category tensors."""
        anchor_cat = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        positive_cat = torch.tensor([0, 1, 1, 2, 2, 0, 3, 3])
        negative_cats = torch.randint(0, 4, (batch_size, num_negatives))
        return anchor_cat, positive_cat, negative_cats

    def test_returns_scalar(self, normalized_embeddings, categories):
        """Loss should return a scalar tensor."""
        anchor, positive, negatives = normalized_embeddings
        anchor_cat, positive_cat, negative_cats = categories

        loss = infonce_loss_with_multiple_negatives(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            anchor_category=anchor_cat,
            positive_category=positive_cat,
            negative_categories=negative_cats,
        )

        assert loss.dim() == 0
        assert loss.dtype == torch.float32

    def test_loss_is_positive(self, normalized_embeddings, categories):
        """Loss should be non-negative."""
        anchor, positive, negatives = normalized_embeddings
        anchor_cat, positive_cat, negative_cats = categories

        loss = infonce_loss_with_multiple_negatives(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            anchor_category=anchor_cat,
            positive_category=positive_cat,
            negative_categories=negative_cats,
        )

        assert loss.item() >= 0

    def test_perfect_positive_gives_low_loss(self, batch_size, num_negatives):
        """When positive = anchor and negatives are random, loss should be low."""
        torch.manual_seed(42)
        dim = 128
        anchor = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
        positive = anchor.clone()  # Perfect match
        negatives = F.normalize(
            torch.randn(batch_size, num_negatives, dim), p=2, dim=-1
        )
        cats = torch.zeros(batch_size, dtype=torch.long)
        neg_cats = torch.ones(batch_size, num_negatives, dtype=torch.long)

        loss = infonce_loss_with_multiple_negatives(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            anchor_category=cats,
            positive_category=cats,
            negative_categories=neg_cats,
        )

        # With perfect positive match, loss should be relatively low
        assert loss.item() < 1.0

    def test_random_embeddings_give_higher_loss(self, batch_size, num_negatives):
        """Random positive should give higher loss than perfect positive."""
        torch.manual_seed(42)
        dim = 128
        anchor = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
        negatives = F.normalize(
            torch.randn(batch_size, num_negatives, dim), p=2, dim=-1
        )
        cats = torch.zeros(batch_size, dtype=torch.long)
        neg_cats = torch.ones(batch_size, num_negatives, dtype=torch.long)

        # Perfect positive
        loss_perfect = infonce_loss_with_multiple_negatives(
            anchor=anchor,
            positive=anchor.clone(),
            negatives=negatives,
            anchor_category=cats,
            positive_category=cats,
            negative_categories=neg_cats,
        )

        # Random positive
        random_positive = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
        loss_random = infonce_loss_with_multiple_negatives(
            anchor=anchor,
            positive=random_positive,
            negatives=negatives,
            anchor_category=cats,
            positive_category=cats,
            negative_categories=neg_cats,
        )

        assert loss_random.item() > loss_perfect.item()

    def test_temperature_scaling(self, normalized_embeddings, categories):
        """Lower temperature should produce higher loss magnitude for random inputs."""
        anchor, positive, negatives = normalized_embeddings
        anchor_cat, positive_cat, negative_cats = categories

        loss_high_temp = infonce_loss_with_multiple_negatives(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            anchor_category=anchor_cat,
            positive_category=positive_cat,
            negative_categories=negative_cats,
            temperature=1.0,
        )

        loss_low_temp = infonce_loss_with_multiple_negatives(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            anchor_category=anchor_cat,
            positive_category=positive_cat,
            negative_categories=negative_cats,
            temperature=0.01,
        )

        # Different temperatures should produce different losses
        assert loss_high_temp.item() != loss_low_temp.item()

    def test_category_weight_effect(self, normalized_embeddings, categories):
        """Category weight should affect loss when categories match."""
        anchor, positive, negatives = normalized_embeddings
        anchor_cat, positive_cat, negative_cats = categories

        loss_no_cat = infonce_loss_with_multiple_negatives(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            anchor_category=anchor_cat,
            positive_category=positive_cat,
            negative_categories=negative_cats,
            category_weight=0.0,
        )

        loss_with_cat = infonce_loss_with_multiple_negatives(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            anchor_category=anchor_cat,
            positive_category=positive_cat,
            negative_categories=negative_cats,
            category_weight=1.0,
        )

        # With same-category pairs present, category_weight > 0 should differ
        assert loss_no_cat.item() != loss_with_cat.item()

    def test_gradient_flows(self, normalized_embeddings, categories):
        """Gradients should flow through the loss."""
        anchor, positive, negatives = normalized_embeddings
        anchor_cat, positive_cat, negative_cats = categories

        # Make anchor require grad
        anchor = anchor.clone().requires_grad_(True)

        loss = infonce_loss_with_multiple_negatives(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            anchor_category=anchor_cat,
            positive_category=positive_cat,
            negative_categories=negative_cats,
        )

        loss.backward()
        assert anchor.grad is not None
        assert not torch.all(anchor.grad == 0)

    def test_batch_size_one(self):
        """Loss should work with batch_size=1."""
        dim = 128
        anchor = F.normalize(torch.randn(1, dim), p=2, dim=-1)
        positive = F.normalize(torch.randn(1, dim), p=2, dim=-1)
        negatives = F.normalize(torch.randn(1, 5, dim), p=2, dim=-1)

        loss = infonce_loss_with_multiple_negatives(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            anchor_category=torch.tensor([0]),
            positive_category=torch.tensor([0]),
            negative_categories=torch.tensor([[1, 2, 3, 1, 2]]),
        )

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_single_negative(self):
        """Loss should work with num_negatives=1."""
        dim = 128
        batch_size = 4
        anchor = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
        positive = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
        negatives = F.normalize(torch.randn(batch_size, 1, dim), p=2, dim=-1)

        loss = infonce_loss_with_multiple_negatives(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            anchor_category=torch.zeros(batch_size, dtype=torch.long),
            positive_category=torch.zeros(batch_size, dtype=torch.long),
            negative_categories=torch.ones(batch_size, 1, dtype=torch.long),
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
