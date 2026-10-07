import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0, ("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(embedding_dim,embedding_dim)
        self.key = nn.Linear(embedding_dim,embedding_dim)
        self.value = nn.Linear(embedding_dim,embedding_dim)
        self.attention_dropout = nn.Dropout(p=dropout)
        self.output_projection = nn.Linear(embedding_dim,embedding_dim)

    def forward(
        self,
        x,
        return_attention=False
    ):

        batch_size, sequence_length, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        # Move heads before sequence dimension
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Shape:
        # [batch, heads, sequence, head_dim]

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        scores = scores.masked_fill(mask == 0,float("-inf"))
        attention_weights = torch.softmax(scores,dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        output = attention_weights @ V

        # Shape:
        # [batch, heads, sequence, head_dim]

        output = output.transpose(1, 2)
        output = output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )
        
        output = self.output_projection(output)

        if return_attention:
            return output, attention_weights

        return output