import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model_definition import build_model


# Define a custom dataset that slices the continuous text data into sequences.
class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        """
        data: numpy array of token IDs.
        seq_length: fixed length of each sequence.
        The target is the input sequence shifted by one character.
        """
        self.data = data
        self.seq_length = seq_length
        # Compute how many sequences we can extract.
        self.num_sequences = (len(data) - 1) // seq_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        # Input sequence: positions [start_idx, end_idx)
        x = self.data[start_idx: end_idx]
        # Target sequence: positions [start_idx+1, end_idx+1)
        y = self.data[start_idx + 1: end_idx + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_processed_data(data_path="processed_data"):
    train_data = np.load(os.path.join(data_path, "train.npy"))
    val_data = np.load(os.path.join(data_path, "val.npy"))
    return train_data, val_data


def load_vocab(data_path="processed_data"):
    vocab_file = os.path.join(data_path, "vocab.pkl")
    with open(vocab_file, "rb") as f:
        vocab, char2idx, idx2char = pickle.load(f)
    return vocab, char2idx, idx2char


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        # Forward pass: outputs shape (batch, seq_length, vocab_size)
        outputs, _ = model(inputs)
        # Flatten outputs and targets for loss computation.
        outputs = outputs.view(-1, outputs.size(2))
        targets = targets.view(-1)
        # Use log of outputs because our model applies softmax already.
        loss = criterion(torch.log(outputs + 1e-8), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, _ = model(inputs)
            outputs = outputs.view(-1, outputs.size(2))
            targets = targets.view(-1)
            loss = criterion(torch.log(outputs + 1e-8), targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    # Hyperparameters
    seq_length = 100
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001

    # Device configuration.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load processed training and validation data.
    train_data, val_data = load_processed_data()

    # Load the vocabulary to determine vocab size.
    vocab, char2idx, idx2char = load_vocab()
    vocab_size = len(vocab)
    print("Vocabulary size:", vocab_size)

    # Create datasets and dataloaders.
    train_dataset = TextDataset(train_data, seq_length)
    val_dataset = TextDataset(val_data, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build the model with the correct vocabulary size.
    model = build_model(vocab_size)
    model = model.to(device)

    # Define the loss function and optimizer.
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop.
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Optionally save the trained model.
    torch.save(model.state_dict(), "char_rnn_model.pth")
    print("Model saved as char_rnn_model.pth")


if __name__ == "__main__":
    main()
