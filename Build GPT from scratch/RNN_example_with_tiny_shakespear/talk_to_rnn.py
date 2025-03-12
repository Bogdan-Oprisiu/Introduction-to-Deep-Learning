import os
import pickle

import numpy as np
import torch

from model_definition import build_model


def load_vocab(data_path="processed_data"):
    vocab_file = os.path.join(data_path, "vocab.pkl")
    with open(vocab_file, "rb") as f:
        vocab, char2idx, idx2char = pickle.load(f)
    return vocab, char2idx, idx2char


def generate_text(model, prompt, char2idx, idx2char, device, generation_length=200):
    """
    Generate text using the model given an initial prompt.
    The function converts the prompt to indices, then in a loop
    it feeds the current sequence to the model and samples the next character.
    """
    # Convert prompt to a list of indices (default to index 0 if character not found)
    input_ids = [char2idx.get(ch, 0) for ch in prompt]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(generation_length):
            # Run the model on the current input sequence
            output, hidden = model(input_tensor)
            # Get the probabilities for the last character in the sequence
            last_probs = output[0, -1, :].cpu().numpy()
            # Sample the next character index from the probability distribution
            next_idx = np.random.choice(len(last_probs), p=last_probs)
            # Append the predicted index to the sequence
            input_ids.append(next_idx)
            # Update the input tensor with the new sequence
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Convert the full sequence of indices back to characters
    generated_text = "".join([idx2char[idx] for idx in input_ids])
    return generated_text


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the vocabulary and mappings
    vocab, char2idx, idx2char = load_vocab()
    vocab_size = len(vocab)
    print("Vocabulary size:", vocab_size)

    # Build the model and load trained weights
    model = build_model(vocab_size)
    model_path = "char_rnn_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print("Loaded trained model from", model_path)
    else:
        print("Trained model not found at", model_path)
        return

    # Interactive loop to chat with the model
    print("Enter a prompt for the model (or 'quit' to exit):")
    while True:
        prompt = input(">> ")
        if prompt.lower() == "quit":
            break
        generated = generate_text(model, prompt, char2idx, idx2char, device, generation_length=200)
        print("\nGenerated text:")
        print(generated)
        print("-" * 40)


if __name__ == "__main__":
    main()
