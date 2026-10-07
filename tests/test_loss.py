import torch

from model.loss import GPTLoss


class TestGPTLoss:

    def test_loss_returns_scalar(self):

        batch_size = 4
        context_length = 128
        vocab_size = 8000

        logits = torch.randn(
            batch_size,
            context_length,
            vocab_size
        )

        targets = torch.randint(
            low=0,
            high=vocab_size,
            size=(
                batch_size,
                context_length
            )
        )

        loss_fn = GPTLoss()

        loss = loss_fn(
            logits,
            targets
        )

        assert loss.ndim == 0

    def test_loss_is_finite(self):

        logits = torch.randn(
            4,
            128,
            8000
        )

        targets = torch.randint(
            0,
            8000,
            (4, 128)
        )

        loss_fn = GPTLoss()

        loss = loss_fn(
            logits,
            targets
        )

        assert torch.isfinite(loss)