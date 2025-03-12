import os
import pickle

import numpy as np
import pandas as pd


def load_csv_text(filename):
    """
    Load a CSV file and assume it has a 'text' column.
    Concatenate all rows into a single string.
    """
    df = pd.read_csv(filename)
    # Ensure the text column is a string and join all lines together.
    text = "\n".join(df['text'].astype(str).tolist())
    return text


def build_vocab(text):
    """
    Build sorted vocabulary of unique characters from the text.
    Create mapping dictionaries: char2idx and idx2char.
    """
    vocab = sorted(set(text))
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for i, ch in enumerate(vocab)}
    return vocab, char2idx, idx2char


def text_to_int(text, char2idx):
    """
    Convert the entire text into a numpy array of integers using the char2idx mapping.
    """
    return np.array([char2idx[ch] for ch in text], dtype=np.int32)


def prepare_data(train_file, val_file, test_file, output_dir="processed_data"):
    # Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    print("Loading CSV files...")
    train_text = load_csv_text(train_file)
    val_text = load_csv_text(val_file)
    test_text = load_csv_text(test_file)

    print("Building vocabulary from training data...")
    vocab, char2idx, idx2char = build_vocab(train_text)

    # Save the vocabulary and mappings (using pickle)
    vocab_file = os.path.join(output_dir, "vocab.pkl")
    with open(vocab_file, "wb") as f:
        pickle.dump((vocab, char2idx, idx2char), f)
    print(f"Vocabulary saved to {vocab_file}")

    # Encode the texts into integer sequences
    train_ids = text_to_int(train_text, char2idx)
    val_ids = text_to_int(val_text, char2idx)
    test_ids = text_to_int(test_text, char2idx)

    # Save the processed data as .npy files
    np.save(os.path.join(output_dir, "train.npy"), train_ids)
    np.save(os.path.join(output_dir, "val.npy"), val_ids)
    np.save(os.path.join(output_dir, "test.npy"), test_ids)
    print(f"Processed data saved in directory: {output_dir}")


if __name__ == "__main__":
    # Update the paths if needed.
    train_file = "train.csv"
    val_file = "validation.csv"
    test_file = "test.csv"

    prepare_data(train_file, val_file, test_file)
