"""Tests for BPR loss functions."""

import pytest
import torch

from yamyam_lab.loss.bpr_loss import bpr_loss, bpr_loss_sampled


class TestBPRLoss:
    """Test basic BPR loss function."""

    def test_bpr_loss_basic(self):
        """Test basic BPR loss calculation."""
        pred_pos = torch.tensor([0.8, 0.9, 0.7])
        pred_neg = torch.tensor([0.3, 0.2, 0.4])

        loss = bpr_loss(pred_pos, pred_neg)

        assert loss.item() > 0, "BPR loss should be positive"
        assert torch.isfinite(loss), "BPR loss should be finite"

    def test_bpr_loss_gradient(self):
        """Test that gradients flow through BPR loss."""
        pred_pos = torch.tensor([0.8, 0.9], requires_grad=True)
        pred_neg = torch.tensor([0.3, 0.2], requires_grad=True)

        loss = bpr_loss(pred_pos, pred_neg)
        loss.backward()

        assert pred_pos.grad is not None, "Positive predictions should have gradients"
        assert pred_neg.grad is not None, "Negative predictions should have gradients"

    def test_bpr_loss_ordering(self):
        """Test that higher positive scores result in lower loss."""
        pred_neg = torch.tensor([0.3, 0.3, 0.3])

        # Case 1: Lower positive scores
        pred_pos_low = torch.tensor([0.4, 0.4, 0.4])
        loss_low = bpr_loss(pred_pos_low, pred_neg)

        # Case 2: Higher positive scores
        pred_pos_high = torch.tensor([0.9, 0.9, 0.9])
        loss_high = bpr_loss(pred_pos_high, pred_neg)

        assert loss_high < loss_low, "Higher positive scores should yield lower loss"


class TestBPRLossSampled:
    """Test sampled BPR loss function with user-wise comparison."""

    def test_basic_functionality(self):
        """Test basic sampled BPR loss."""
        scores = torch.tensor([0.8, 0.9, 0.3, 0.2, 0.7, 0.4])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

        loss = bpr_loss_sampled(scores, targets)

        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_user_wise_comparison(self):
        """Test user-wise BPR loss calculation."""
        # User 0: 1 pos, 3 neg
        # User 1: 2 pos, 2 neg
        user_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        targets = torch.tensor([1, 0, 0, 0, 1, 1, 0, 0]).float()
        scores = torch.tensor([0.8, 0.2, 0.3, 0.1, 0.9, 0.7, 0.4, 0.2])

        loss = bpr_loss_sampled(scores, targets, user_ids=user_ids)

        assert loss.item() > 0, "User-wise loss should be positive"
        assert torch.isfinite(loss), "User-wise loss should be finite"

    def test_cross_user_negatives(self):
        """Test that cross-user negatives are used when needed."""
        # User 0: 1 pos, 0 neg (should use User 1's negatives)
        # User 1: 0 pos, 10 neg
        user_ids = torch.tensor([0] + [1] * 10)
        targets = torch.tensor([1.0] + [0.0] * 10)
        scores = torch.randn(11, requires_grad=True)

        loss = bpr_loss_sampled(scores, targets, user_ids=user_ids)

        assert loss.item() > 0, "Should compute loss using cross-user negatives"
        assert scores.grad is None  # Not backpropagated yet

        loss.backward()
        assert scores.grad is not None, (
            "Gradients should flow through cross-user negatives"
        )

    def test_extreme_imbalance(self):
        """Test handling of extreme positive:negative imbalance (1:99)."""
        batch_size = 100
        num_pos = 1
        num_neg = 99

        targets = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)]).float()
        user_ids = torch.randint(0, 10, (batch_size,))
        scores = torch.randn(batch_size, requires_grad=True)

        loss = bpr_loss_sampled(scores, targets, user_ids=user_ids, sample_negatives=10)

        assert loss.item() > 0, "Should handle extreme imbalance"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_negative_sampling(self):
        """Test that negative sampling limits the number of comparisons."""
        # One user with 1 pos and 100 negs
        user_ids = torch.zeros(101, dtype=torch.long)
        targets = torch.cat([torch.ones(1), torch.zeros(100)]).float()
        scores = torch.randn(101, requires_grad=True)

        # Sample only 10 negatives
        loss = bpr_loss_sampled(scores, targets, user_ids=user_ids, sample_negatives=10)

        assert loss.item() > 0, "Should compute loss with sampled negatives"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_all_positives_edge_case(self):
        """Test edge case with all positive samples."""
        targets = torch.ones(50).float()
        scores = torch.randn(50)
        user_ids = torch.randint(0, 5, (50,))

        loss = bpr_loss_sampled(scores, targets, user_ids=user_ids)

        assert loss.item() == 0.0, "All positives should yield zero loss"

    def test_all_negatives_edge_case(self):
        """Test edge case with all negative samples."""
        targets = torch.zeros(50).float()
        scores = torch.randn(50)
        user_ids = torch.randint(0, 5, (50,))

        loss = bpr_loss_sampled(scores, targets, user_ids=user_ids)

        assert loss.item() == 0.0, "All negatives should yield zero loss"

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the loss."""
        user_ids = torch.randint(0, 20, (50,))
        targets = torch.cat([torch.ones(5), torch.zeros(45)]).float()
        scores = torch.randn(50, requires_grad=True)

        loss = bpr_loss_sampled(scores, targets, user_ids=user_ids, sample_negatives=10)

        assert loss.item() > 0, "Loss should be positive"

        loss.backward()

        assert scores.grad is not None, "Gradients should exist"
        assert torch.isfinite(scores.grad).all(), "Gradients should be finite"
        assert scores.grad.norm().item() > 0, "Gradient norm should be positive"

    def test_debug_mode(self):
        """Test debug mode output (doesn't crash)."""
        user_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        targets = torch.tensor([1, 0, 0, 1, 0, 0]).float()
        scores = torch.randn(6)

        # Should not crash with debug=True
        loss = bpr_loss_sampled(scores, targets, user_ids=user_ids, debug=True)

        assert loss.item() >= 0, "Loss should be non-negative"

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with fixed seed."""
        user_ids = torch.zeros(101, dtype=torch.long)
        targets = torch.cat([torch.ones(1), torch.zeros(100)]).float()
        scores = torch.randn(101)  # Fixed scores for both runs

        # Run 1: Compute loss with seed set
        torch.manual_seed(42)
        loss1 = bpr_loss_sampled(
            scores, targets, user_ids=user_ids, sample_negatives=10
        )

        # Run 2: Same seed should give same loss (deterministic sampling)
        torch.manual_seed(42)
        loss2 = bpr_loss_sampled(
            scores, targets, user_ids=user_ids, sample_negatives=10
        )

        # With same seed, loss should be deterministic
        assert torch.isclose(loss1, loss2, rtol=1e-5), (
            f"Same seed should give same loss: {loss1.item():.6f} vs {loss2.item():.6f}"
        )

    @pytest.mark.parametrize("sample_size", [1, 5, 10, 50])
    def test_different_sample_sizes(self, sample_size):
        """Test different negative sample sizes."""
        user_ids = torch.zeros(101, dtype=torch.long)
        targets = torch.cat([torch.ones(1), torch.zeros(100)]).float()
        scores = torch.randn(101, requires_grad=True)

        loss = bpr_loss_sampled(
            scores, targets, user_ids=user_ids, sample_negatives=sample_size
        )

        assert loss.item() > 0, (
            f"Loss should be positive with sample_size={sample_size}"
        )
        assert torch.isfinite(loss), (
            f"Loss should be finite with sample_size={sample_size}"
        )
