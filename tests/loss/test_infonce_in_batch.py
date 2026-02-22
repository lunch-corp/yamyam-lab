"""Tests for in-batch InfoNCE loss and positive mask builder."""

import torch
import torch.nn.functional as F

from yamyam_lab.loss.infonce import build_positive_mask, infonce_in_batch_loss


class TestBuildPositiveMask:
    """Tests for build_positive_mask."""

    def test_shape(self):
        """Mask shape is (B, B)."""
        anchor = torch.tensor([0, 1, 2, 3])
        positive = torch.tensor([10, 11, 12, 13])
        d2p = {}
        mask = build_positive_mask(anchor, positive, d2p)
        assert mask.shape == (4, 4)

    def test_diagonal_never_masked(self):
        """Diagonal entries are always False even when pair is known positive."""
        anchor = torch.tensor([0, 1])
        positive = torch.tensor([10, 11])
        d2p = {0: {10, 11}, 1: {10, 11}}
        mask = build_positive_mask(anchor, positive, d2p)
        assert not mask[0, 0]
        assert not mask[1, 1]

    def test_known_positive_masked(self):
        """Known positive off-diagonal entries are True."""
        anchor = torch.tensor([0, 1])
        positive = torch.tensor([10, 11])
        # Diner 0 has positive 11, which is at position j=1
        d2p = {0: {11}}
        mask = build_positive_mask(anchor, positive, d2p)
        assert mask[0, 1]  # anchor 0's known positive 11 is at j=1
        assert not mask[1, 0]  # diner 1 has no known positives

    def test_no_known_positives_all_false(self):
        """Empty diner_to_positives produces all-False mask."""
        anchor = torch.tensor([0, 1, 2])
        positive = torch.tensor([10, 11, 12])
        d2p = {}
        mask = build_positive_mask(anchor, positive, d2p)
        assert not mask.any()

    def test_all_positives_masked_except_diagonal(self):
        """When all diners are mutual positives, only diagonal is False."""
        anchor = torch.tensor([0, 1, 2])
        positive = torch.tensor([0, 1, 2])
        d2p = {0: {0, 1, 2}, 1: {0, 1, 2}, 2: {0, 1, 2}}
        mask = build_positive_mask(anchor, positive, d2p)
        for i in range(3):
            assert not mask[i, i]
            for j in range(3):
                if i != j:
                    assert mask[i, j]


class TestInfoNCEInBatchLoss:
    """Tests for infonce_in_batch_loss."""

    def test_returns_scalar(self):
        """Loss is a scalar tensor."""
        B, D = 8, 32
        anchor = F.normalize(torch.randn(B, D), dim=-1)
        positive = F.normalize(torch.randn(B, D), dim=-1)
        mask = torch.zeros(B, B, dtype=torch.bool)
        loss = infonce_in_batch_loss(anchor, positive, mask)
        assert loss.dim() == 0

    def test_loss_non_negative(self):
        """Loss is non-negative."""
        B, D = 8, 32
        anchor = F.normalize(torch.randn(B, D), dim=-1)
        positive = F.normalize(torch.randn(B, D), dim=-1)
        mask = torch.zeros(B, B, dtype=torch.bool)
        loss = infonce_in_batch_loss(anchor, positive, mask)
        assert loss.item() >= 0

    def test_without_mask_equals_standard_infonce(self):
        """With all-False mask, loss matches standard CLIP-style InfoNCE."""
        B, D = 4, 16
        torch.manual_seed(42)
        anchor = F.normalize(torch.randn(B, D), dim=-1)
        positive = F.normalize(torch.randn(B, D), dim=-1)

        mask = torch.zeros(B, B, dtype=torch.bool)
        loss = infonce_in_batch_loss(anchor, positive, mask, temperature=0.07)

        # Manual computation
        sim = anchor @ positive.T / 0.07
        labels = torch.arange(B)
        expected = F.cross_entropy(sim, labels)

        assert torch.allclose(loss, expected, atol=1e-5)

    def test_perfect_match_low_loss(self):
        """When positive==anchor and no mask, loss should be very low."""
        B, D = 8, 32
        torch.manual_seed(42)
        anchor = F.normalize(torch.randn(B, D), dim=-1)
        positive = anchor.clone()
        mask = torch.zeros(B, B, dtype=torch.bool)
        loss = infonce_in_batch_loss(anchor, positive, mask, temperature=0.07)
        assert loss.item() < 1.0

    def test_masked_entries_reduce_loss(self):
        """Masking false negatives should reduce or change the loss."""
        B, D = 8, 32
        torch.manual_seed(42)
        anchor = F.normalize(torch.randn(B, D), dim=-1)
        positive = F.normalize(torch.randn(B, D), dim=-1)

        no_mask = torch.zeros(B, B, dtype=torch.bool)
        loss_no_mask = infonce_in_batch_loss(anchor, positive, no_mask)

        # Create mask that removes some false negatives
        with_mask = torch.zeros(B, B, dtype=torch.bool)
        with_mask[0, 1] = True
        with_mask[1, 0] = True
        loss_with_mask = infonce_in_batch_loss(anchor, positive, with_mask)

        # Losses should differ when mask is applied
        assert loss_no_mask.item() != loss_with_mask.item()

    def test_gradient_flows(self):
        """Gradients flow through the loss."""
        B, D = 4, 16
        anchor = F.normalize(torch.randn(B, D), dim=-1).requires_grad_(True)
        positive = F.normalize(torch.randn(B, D), dim=-1)
        mask = torch.zeros(B, B, dtype=torch.bool)
        loss = infonce_in_batch_loss(anchor, positive, mask)
        loss.backward()
        assert anchor.grad is not None
        assert not torch.all(anchor.grad == 0)

    def test_temperature_effect(self):
        """Different temperatures produce different losses."""
        B, D = 4, 16
        torch.manual_seed(42)
        anchor = F.normalize(torch.randn(B, D), dim=-1)
        positive = F.normalize(torch.randn(B, D), dim=-1)
        mask = torch.zeros(B, B, dtype=torch.bool)

        loss_high = infonce_in_batch_loss(anchor, positive, mask, temperature=1.0)
        loss_low = infonce_in_batch_loss(anchor, positive, mask, temperature=0.01)
        assert loss_high.item() != loss_low.item()

    def test_batch_size_one(self):
        """Loss works with batch_size=1."""
        anchor = F.normalize(torch.randn(1, 16), dim=-1)
        positive = F.normalize(torch.randn(1, 16), dim=-1)
        mask = torch.zeros(1, 1, dtype=torch.bool)
        loss = infonce_in_batch_loss(anchor, positive, mask)
        assert loss.dim() == 0
        assert loss.item() >= 0
