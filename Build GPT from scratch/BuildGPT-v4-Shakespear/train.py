import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tokenizer import CharTokenizer
from model import GPT

#####################
# 1. Hyperparameters
#####################
SEQ_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3  # learning rate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


#######################
# 2. Custom Dataset
#######################
class ShakespeareDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Encode all texts into a single list of tokens
        self.tokens = []
        for t in texts:
            self.tokens.extend(self.tokenizer.encode(t))

        # Create input-target pairs (shift targets by 1)
        self.inputs = []
        self.targets = []
        for i in range(len(self.tokens) - seq_len):
            self.inputs.append(self.tokens[i: i + seq_len])
            self.targets.append(self.tokens[i + 1: i + seq_len + 1])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.long)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y


#######################
# 3. Load CSV Datasets
#######################
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # We assume the CSV has a column named "text"
    texts = df["text"].tolist()
    return texts


train_texts = load_dataset("train.csv")
val_texts = load_dataset("validation.csv")
test_texts = load_dataset("test.csv")

###############################
# 4. Build/Load Tokenizer
###############################
tokenizer = CharTokenizer()
# Build vocab from train data only
tokenizer.build_vocab(" ".join(train_texts))

###############################
# 5. Create DataLoaders
###############################
train_dataset = ShakespeareDataset(train_texts, tokenizer, SEQ_LEN)
val_dataset = ShakespeareDataset(val_texts, tokenizer, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

vocab_size = len(tokenizer.vocab)
print(f"Vocab size: {vocab_size}")

###############################
# 6. Initialize Model + Optim
###############################
model = GPT(
    vocab_size=vocab_size,
    embed_dim=256,  # Increase if you have enough GPU memory
    num_heads=4,  # Adjust
    num_layers=4,  # Adjust
    seq_len=SEQ_LEN,
    dropout=0.1
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


#############################################
# 7. Training and Validation Loop
#############################################
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0

    for batch in loader:
        x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        # Forward
        # logits shape: [batch_size, seq_len, vocab_size]
        logits = model(x)

        # Flatten for CrossEntropy: [batch_size * seq_len, vocab_size]
        logits = logits.reshape(-1, vocab_size)
        y = y.reshape(-1)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Forward
            logits = model(x)
            logits = logits.reshape(-1, vocab_size)
            y = y.reshape(-1)

            loss = criterion(logits, y)
            total_loss += loss.item()

    return total_loss / len(loader)


for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    ###############################
    # 8. Test Performance
    ###############################
    test_dataset = ShakespeareDataset(test_texts, tokenizer, SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    test_loss = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    ###############################
    # 9. Save the Model
    ###############################
    torch.save(model.state_dict(), "gpt_shakespeare.pth")
    print("Model saved!")
