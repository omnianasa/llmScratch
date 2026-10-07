import torch

from model.feed_forward import FeedForward


class TestFeedForward:

    def test_feed_forward_shape(self):

        batch_size = 32
        context_length = 128
        embedding_dim = 256
        hidden_dim = 1024

        model = FeedForward(
            embedding_dim=embedding_dim,
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