import math
import torch
import torch.nn as nn
import pickle

# ------------------------------
# Model and Positional Encoding
# ------------------------------

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
        # Using the same tensor for memory (GPT-style self-attention)
        output = self.transformer_decoder(tgt_emb, tgt_emb, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        logits = self.fc_out(output)
        return logits

# ------------------------------
# Vocabulary and Tokenizer
# ------------------------------
# These helper functions convert between text and token IDs.
# They rely on vocabulary mappings saved in 'vocab.pkl' (a dictionary with 'stoi' and 'itos').

def load_vocab(vocab_file="vocab.pkl"):
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    return vocab["stoi"], vocab["itos"]

def encode(text, stoi):
    # A simple whitespace tokenizer; adjust if you used a more advanced tokenizer.
    tokens = text.split()
    token_ids = [stoi.get(token, 0) for token in tokens]  # 0 is <unk>
    return token_ids

def decode(token_ids, itos):
    tokens = [itos[i] for i in token_ids]
    return " ".join(tokens)

# ------------------------------
# Text Generation Function
# ------------------------------

def generate_text(model, prompt, stoi, itos, max_length=50):
    model.eval()
    # Encode the prompt
    input_ids = encode(prompt, stoi)
    generated = input_ids.copy()

    # Convert to tensor and move to device
    input_tensor = torch.tensor([generated], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_tensor)  # (batch, seq_len, vocab_size)
            # Get the logits for the last token and apply greedy sampling
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            generated.append(next_token)
            # Update input_tensor with the newly generated token
            input_tensor = torch.tensor([generated], dtype=torch.long).to(device)
    return decode(generated, itos)

# ------------------------------
# Main Inference Code (Interactive Chat)
# ------------------------------

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using GPU for inference.")
    else:
        print("Using CPU for inference.")

    # Load vocabulary mappings
    try:
        stoi, itos = load_vocab("vocab.pkl")
    except FileNotFoundError:
        print("Vocabulary file 'vocab.pkl' not found. Make sure to save your vocabulary during training.")
        exit(1)
    actual_vocab_size = len(itos)
    print(f"Vocabulary size: {actual_vocab_size}")

    # Initialize and load the trained model
    model = GPTSmall(actual_vocab_size).to(device)
    model_state = torch.load("gpt_small.pth", map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    print("Model loaded successfully. You can start chatting now.")

    # Interactive loop
    print("Enter 'quit' to exit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == "quit":
            break
        response = generate_text(model, prompt, stoi, itos, max_length=50)
        print("GPT:", response)
