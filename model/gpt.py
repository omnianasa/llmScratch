import torch.nn as nn

from model.embedding import TokenEmbedding
from model.positional_embedding import PositionalEmbedding
from model.transformer_block import TransformerBlock


class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

        self.positional_embedding = PositionalEmbedding(
            context_length=context_length,
            embedding_dim=embedding_dim
        )

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final Normalization
        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

        # Language Model Head
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        # Token embedding
        x = self.token_embedding(x)

        # Positional embedding
        x = self.positional_embedding(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Vocabulary logits
        logits = self.lm_head(x)

        return logits