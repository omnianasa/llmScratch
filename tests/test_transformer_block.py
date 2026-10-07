import torch

from model.transformer_block import TransformerBlock


class TestTransformerBlock:

    def test_transformer_block_shape(self):

        batch_size = 32
        context_length = 128
        embedding_dim = 256
        num_heads = 4
        hidden_dim = 1024

        model = TransformerBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim
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