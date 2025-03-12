import os
import pickle

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, conv_channels=64, rnn_hidden_size=256, rnn_layers=1):
        """
        vocab_size: number of unique characters in the vocabulary.
        embedding_dim: dimension of the character embeddings.
        conv_channels: number of output channels from the convolutional layer.
        rnn_hidden_size: hidden size for the LSTM layer.
        rnn_layers: number of recurrent layers.
        """
        super(CharRNN, self).__init__()

        # 1. Embedding layer to convert input indices into dense vectors.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. Normalization: BatchNorm applied to the embedding dimension.
        #    Note: BatchNorm1d expects shape (batch, channels, length)
        self.bn = nn.BatchNorm1d(embedding_dim)

        # 3. Convolutional layer with LeakyReLU activation.
        #    We use kernel_size=3 with padding=1 to preserve the sequence length.
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        # 4. Recurrent layer (LSTM) for sequence modeling.
        #    Input size to the LSTM is conv_channels.
        self.rnn = nn.LSTM(input_size=conv_channels, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                           batch_first=True)

        # 5. Fully connected layer to project LSTM outputs to vocabulary size.
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

        # 6. Softmax layer to produce probabilities over the vocabulary.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hidden=None):
        """
        x: Tensor of shape (batch_size, seq_length) containing input character indices.
        hidden: Optional hidden state for the LSTM.

        Returns:
            x: Tensor of shape (batch_size, seq_length, vocab_size) containing probability distributions.
            hidden: Final hidden state from the LSTM.
        """
        # Embedding: (batch_size, seq_length, embedding_dim)
        x = self.embedding(x)

        # Batch normalization: Permute to (batch_size, embedding_dim, seq_length)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        # Permute back to (batch_size, seq_length, embedding_dim)
        x = x.permute(0, 2, 1)

        # Prepare for convolution: permute to (batch_size, embedding_dim, seq_length)
        x = x.permute(0, 2, 1)
        # Convolution layer: output shape (batch_size, conv_channels, seq_length)
        x = self.conv1(x)
        x = self.leaky_relu(x)

        # Rearrange for LSTM: (batch_size, seq_length, conv_channels)
        x = x.permute(0, 2, 1)

        # LSTM layer: output shape (batch_size, seq_length, rnn_hidden_size)
        if hidden is None:
            x, hidden = self.rnn(x)
        else:
            x, hidden = self.rnn(x, hidden)

        # Fully connected projection to vocab size: (batch_size, seq_length, vocab_size)
        x = self.fc(x)

        # Apply softmax to get probabilities for each character
        x = self.softmax(x)
        return x, hidden


def build_model(vocab_size, **kwargs):
    """
    Helper function to instantiate the model with a given vocabulary size and optional hyperparameters.
    """
    return CharRNN(vocab_size, **kwargs)


if __name__ == "__main__":
    # Load the vocabulary from the output folder of your data setup file.
    vocab_path = os.path.join("processed_data", "vocab.pkl")
    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            vocab, char2idx, idx2char = pickle.load(f)
        vocab_size = len(vocab)
        print("Vocabulary size loaded from file:", vocab_size)
    else:
        print("Vocabulary file not found. Please run data_setup.py first.")
        vocab_size = 0

    if vocab_size > 0:
        model = build_model(vocab_size)
        # Create a dummy input with shape (batch_size, seq_length)
        dummy_input = torch.randint(0, vocab_size, (16, 100)).long()  # random batch of indices
        output, hidden = model(dummy_input)
        print("Output shape:", output.shape)  # Expected: (16, 100, vocab_size)
