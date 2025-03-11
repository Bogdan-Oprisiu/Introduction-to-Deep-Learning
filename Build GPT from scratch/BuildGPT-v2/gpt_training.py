import math
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


#######################################
# The GPT-like model and positional encoding
#######################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GPTSmall(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Create a stack of TransformerDecoderLayers
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        # Create a causal mask to prevent attention to future tokens.
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt):
        # tgt shape: (batch_size, seq_len)
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        # Transformer expects shape: (seq_len, batch_size, d_model)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_emb.device)
        # Using the same tensor for memory (a workaround for GPT-style self-attention)
        output = self.transformer_decoder(tgt_emb, tgt_emb, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        logits = self.fc_out(output)
        return logits


#######################################
# Data Processing
#######################################

def load_text(filename):
    """Read the training data from a file."""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(tokens, vocab_size=10000):
    """Build a vocabulary mapping (token to index and index to token)."""
    counter = Counter(tokens)
    # Reserve index 0 for <unk>
    most_common = counter.most_common(vocab_size - 1)
    itos = ['<unk>'] + [token for token, _ in most_common]
    stoi = {token: i for i, token in enumerate(itos)}
    return stoi, itos


class TextDataset(Dataset):
    def __init__(self, tokens, stoi, seq_len):
        self.seq_len = seq_len
        # Map tokens to indices; unknown tokens get index 0
        self.token_ids = [stoi.get(token, 0) for token in tokens]

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        x = self.token_ids[idx: idx + self.seq_len]
        y = self.token_ids[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


#######################################
# Training Loop
#######################################

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch_inputs, batch_targets in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        logits = model(batch_inputs)  # logits shape: (batch, seq_len, vocab_size)
        # Reshape logits and targets to compute loss
        loss = loss_fn(logits.view(-1, logits.size(-1)), batch_targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


#######################################
# Main Script
#######################################

def main():
    # Hyperparameters
    seq_len = 50  # Length of training sequences
    batch_size = 32
    num_epochs = 10
    lr = 1e-3
    vocab_size = 10000  # Maximum vocabulary size

    # Check the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using GPU for training.")
    else:
        print("Using CPU for training.")

    # Load and process training data
    if not os.path.exists("input.txt"):
        print("Error: 'input.txt' not found. Please prepare your training data in input.txt")
        return

    raw_text = load_text("input.txt")
    tokens = raw_text.split()  # Simple whitespace tokenizer

    # After building the vocabulary:
    stoi, itos = build_vocab(tokens, vocab_size)
    actual_vocab_size = len(itos)
    print(f"Vocabulary size: {actual_vocab_size}")

    # Save vocabulary mappings
    import pickle
    with open("vocab.pkl", "wb") as f:
        pickle.dump({"stoi": stoi, "itos": itos}, f)
    print("Vocabulary saved to vocab.pkl")

    dataset = TextDataset(tokens, stoi, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model with the actual vocabulary size
    model = GPTSmall(actual_vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Print training start message (before first epoch)
    print("Training has started...")

    # Training loop
    for epoch in range(1, num_epochs + 1):
        avg_loss = train(model, dataloader, optimizer, loss_fn, device)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    # Save the trained model
    model_file = "gpt_small.pth"
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")


if __name__ == "__main__":
    main()
