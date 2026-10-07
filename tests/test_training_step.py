import torch

from model.gpt import GPTModel
from model.loss import GPTLoss


class TestTrainingStep:

    def test_training_step(self):

        vocab_size = 8000
        context_length = 128
        embedding_dim = 256

        model = GPTModel(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim,
            num_heads=4,
            hidden_dim=1024,
            num_layers=4
        )

        loss_fn = GPTLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4
        )

        x = torch.randint(
            0,
            vocab_size,
            (4, context_length)
        )

        targets = torch.randint(
            0,
            vocab_size,
            (4, context_length)
        )

        # Save one parameter before update
        parameter_before = (
            model.token_embedding.embedding.weight
            .detach()
            .clone()
        )

        # Training step
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits,targets)
        loss.backward()
        optimizer.step()

        # Checks
        assert torch.isfinite(loss)

        parameter_after = (
            model.token_embedding.embedding.weight
            .detach()
            .clone()
        )

        assert not torch.equal(
            parameter_before,
            parameter_after
        )