import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, context_length: int, embedding_dim: int):
        super().__init__()

        self.position_embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape

        positions = torch.arange(
            sequence_length,
            device=x.device,
        )

        position_embeddings = self.position_embedding(positions)

        return x + position_embeddings