import torch

from model.attention.self_attention import SelfAttention


class TestSelfAttention:

    def test_self_attention_shape(self):
        batch_size = 32
        context_length = 128
        embedding_dim = 256

        model = SelfAttention(embedding_dim=embedding_dim)

        x = torch.randn(
            batch_size,
            context_length,
            embedding_dim
        )

        output = model(x)

        assert output.shape == (
            batch_size,
            context_length,
            embedding_dim
        )