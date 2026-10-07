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

    def test_attention_weights_shape(self):
        batch_size = 1
        context_length = 4
        embedding_dim = 8

        model = SelfAttention(
            embedding_dim=embedding_dim
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
            context_length,
            context_length
        )

    def test_causal_mask(self):

        batch_size = 1
        context_length = 4
        embedding_dim = 8

        model = SelfAttention(
            embedding_dim=embedding_dim
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

        # Future tokens must receive zero attention

        assert torch.all(
            attention_weights[0].triu(diagonal=1) == 0
        )