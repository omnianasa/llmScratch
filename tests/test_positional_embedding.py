import torch

from model.positional_embedding import PositionalEmbedding


class TestPositionalEmbedding:

    def test_positional_embedding_shape(self):
        batch_size = 32
        context_length = 128
        embedding_dim = 256

        model = PositionalEmbedding(
            context_length=context_length,
            embedding_dim=embedding_dim,
        )

        x = torch.randn(
            batch_size,
            context_length,
            embedding_dim,
        )

        output = model(x)

        assert output.shape == (
            batch_size,
            context_length,
            embedding_dim,
        )

    def test_position_embedding_shape(self):
        context_length = 128
        embedding_dim = 256

        model = PositionalEmbedding(
            context_length=context_length,
            embedding_dim=embedding_dim,
        )

        assert model.position_embedding.weight.shape == (
            context_length,
            embedding_dim,
        )