import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.query = nn.Linear(embedding_dim,embedding_dim)
        self.key = nn.Linear(embedding_dim,embedding_dim)
        self.value = nn.Linear(embedding_dim,embedding_dim)

    def forward(self, x, return_attention=False):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(K.size(-1))

        # Causal mask
        sequence_length = x.size(1)

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        output = attention_weights @ V

        if return_attention:
            return output, attention_weights
        return output