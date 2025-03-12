import pandas as pd
import torch

from tokenizer import CharTokenizer
from model import GPT


##########################
# A) Load Dataset (train)
##########################
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    return texts


##########################
# B) Build the Tokenizer
##########################
def build_same_tokenizer():
    # 1) Read the SAME train.csv used in train.py
    train_texts = load_dataset("train.csv")

    # 2) Create a new tokenizer
    tokenizer = CharTokenizer()

    # 3) Build vocab the same way as in train.py
    #    (train.py uses: tokenizer.build_vocab(\" \".join(train_texts)))
    tokenizer.build_vocab(" ".join(train_texts))

    return tokenizer


##########################
# C) Load Model & Tokenizer
##########################
def load_model(model_path):
    # 1) Rebuild the tokenizer from train.csv
    tokenizer = build_same_tokenizer()
    vocab_size = len(tokenizer.vocab)

    # 2) Recreate the GPT architecture
    model = GPT(
        vocab_size=vocab_size,
        embed_dim=256,  # must match what you used in train.py
        num_heads=4,
        num_layers=4,
        seq_len=128,
        dropout=0.1
    )

    # 3) Load model weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer


##########################
# D) Generation Function
##########################
def generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=50,
        temperature=1.0,
        use_gpu=False
):
    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    model.to(device)
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward
            logits = model(input_ids)  # [1, seq_len, vocab_size]
            last_token_logits = logits[0, -1, :]

            # Temperature
            last_token_logits = last_token_logits / temperature

            # Option A: Greedy
            # next_token_id = torch.argmax(last_token_logits).unsqueeze(0)

            # Option B: Sampling
            probs = torch.softmax(last_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

    # Decode
    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids)


##########################
# E) Console Loop
##########################
def main():
    model_path = "gpt_shakespeare.pth"
    model, tokenizer = load_model(model_path)

    print("Welcome to GPT Console Chat! Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        response = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=100,
            temperature=1.0,
            use_gpu=False
        )

        print("Model:", response)


if __name__ == "__main__":
    main()
