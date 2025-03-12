import torch
import torch.nn as nn
import math


class GPT(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            seq_len=128,
            dropout=0.1
    ):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Positional embedding
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final linear layer for next-token prediction
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        x: [batch_size, seq_len] of token IDs
        Return: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape

        # Create token embeddings
        token_emb = self.token_embedding(x)  # [batch_size, seq_len, embed_dim]

        # Create positional embeddings
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        pos_emb = self.pos_embedding(positions)  # [1, seq_len, embed_dim]

        # Sum token + positional embeddings
        # Transformer expects shape [seq_len, batch_size, embed_dim]
        emb = token_emb + pos_emb
        emb = emb.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]

        # Transformer Encoder
        # We use a causal mask so it cannot see future tokens
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)  # [seq_len, seq_len]
        encoded = self.transformer(emb, mask=mask)  # [seq_len, batch_size, embed_dim]

        # Output layer
        logits = self.fc_out(encoded)  # [seq_len, batch_size, vocab_size]
        logits = logits.permute(1, 0, 2)  # [batch_size, seq_len, vocab_size]

        return logits

    def generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
