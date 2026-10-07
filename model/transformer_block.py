import torch.nn as nn

from model.attention.multi_head_attention import (
    MultiHeadSelfAttention
)

from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Layer Normalization
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)

        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # FNN 
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        # Residual Dropout
        self.residual_dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        # Attention block
        normalized_x = self.attention_norm(x)

        attention_output = self.attention(normalized_x)
        x = x + self.residual_dropout(attention_output)

        # Feed-Forward block
        normalized_x = self.feed_forward_norm(x)
        feed_forward_output = self.feed_forward(normalized_x)
        x = x + self.residual_dropout(feed_forward_output)

        return x