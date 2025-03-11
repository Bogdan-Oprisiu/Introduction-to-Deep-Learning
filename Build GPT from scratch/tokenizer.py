import json
import pickle
from typing import List


class CharTokenizer:
    def __init__(self, text: str = None, normalize_whitespace: bool = False):
        """Initialize the tokenizer. If text is provided, builds a vocabulary."""
        self.vocab = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.normalize_whitespace = normalize_whitespace
        if text:
            self.build_vocab(text)

    def build_vocab(self, text: str):
        """Builds a character-level vocabulary from the given text."""
        if self.normalize_whitespace:
            text = ' '.join(text.split())  # Normalize whitespace
        self.vocab = sorted(set(text))  # Get unique characters
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def encode(self, text: str) -> List[int]:
        """Converts text into a list of token IDs."""
        if self.normalize_whitespace:
            text = ' '.join(text.split())  # Normalize whitespace before encoding
        if not self.char_to_idx:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def decode(self, tokens: List[int]) -> str:
        """Converts token IDs back to text."""
        if not self.idx_to_char:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        return ''.join(self.idx_to_char[token] for token in tokens)

    def save(self, file_path: str):
        """Saves the tokenizer vocabulary to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f)

        # Save vocabulary as a pickle file
        with open("vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)

    def load(self, file_path: str):
        """Loads the tokenizer vocabulary from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.build_vocab(''.join(data["vocab"]))

        # Load vocabulary from a pickle file (if available)
        try:
            with open("vocab.pkl", "rb") as f:
                self.vocab = pickle.load(f)
        except FileNotFoundError:
            pass


# Example Usage
if __name__ == "__main__":
    sample_text = "To be, or not to be, that is the question."
    tokenizer = CharTokenizer(sample_text, normalize_whitespace=True)
    encoded = tokenizer.encode("To be")
    decoded = tokenizer.decode(encoded)

    print("Encoded:", encoded)  # Example: [10, 5, 1, 2, 5]
    print("Decoded:", decoded)  # Output: "To be"

    tokenizer.save("tokenizer.json")
    new_tokenizer = CharTokenizer()
    new_tokenizer.load("tokenizer.json")
    print("Loaded Vocab:", new_tokenizer.vocab)
