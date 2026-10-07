import torch

from model.embedding import TokenEmbedding


class TestTokenEmbedding:

    def test_embedding_shape(self):
        vocab_size = 8000
        embedding_dim = 256
        batch_size = 32
        context_length = 128

        model = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        x = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, context_length),
        )

        output = model(x)

        assert output.shape == (
            batch_size,
            context_length,
            embedding_dim,
        )