import torch

from model.gpt import GPTModel


class TestGPTModel:

    def test_gpt_output_shape(self):

        batch_size = 4
        context_length = 128
        vocab_size = 8000
        embedding_dim = 256

        model = GPTModel(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim,
            num_heads=4,
            hidden_dim=1024,
            num_layers=4
        )

        x = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, context_length)
        )

        logits = model(x)

        assert logits.shape == (
            batch_size,
            context_length,
            vocab_size
        )