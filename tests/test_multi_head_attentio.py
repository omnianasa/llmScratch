import torch

from model.attention.multi_head_attention import (
    MultiHeadSelfAttention
)


class TestMultiHeadSelfAttention:

    def test_multi_head_attention_shape(self):

        batch_size = 32
        context_length = 128
        embedding_dim = 256
        num_heads = 4

        model = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

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

    def test_attention_weights_shape(self):

        batch_size = 1
        context_length = 4
        embedding_dim = 8
        num_heads = 2

        model = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        x = torch.randn(
            batch_size,
            context_length,
            embedding_dim
        )

        output, attention_weights = model(
            x,
            return_attention=True
        )

        assert output.shape == (
            batch_size,
            context_length,
            embedding_dim
        )

        assert attention_weights.shape == (
            batch_size,
            num_heads,
            context_length,
            context_length
        )

    def test_causal_mask(self):

        batch_size = 1
        context_length = 4
        embedding_dim = 8
        num_heads = 2

        model = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        x = torch.randn(
            batch_size,
            context_length,
            embedding_dim
        )

        _, attention_weights = model(
            x,
            return_attention=True
        )

        future_attention = attention_weights.triu(
            diagonal=1
        )

        assert torch.all(
            future_attention == 0
        )

    def test_head_dimension(self):

        model = MultiHeadSelfAttention(
            embedding_dim=256,
            num_heads=4
        )

        assert model.head_dim == 64