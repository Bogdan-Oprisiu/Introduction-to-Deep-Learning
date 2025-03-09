#!/usr/bin/env python
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define a custom dataset.
class CustomDataset(Dataset):
    def __init__(self, num_samples=1000):
        # Generate random 2D data between -1 and 1.
        self.data = np.random.rand(num_samples, 2) * 2 - 1
        # Simple rule: label is 1 if x + y > 0, else 0.
        self.labels = (self.data[:, 0] + self.data[:, 1] > 0).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample, label


dataset = CustomDataset(num_samples=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define a simple MLP model.
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = SimpleMLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop.
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    for batch_data, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs.squeeze(), batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_data.size(0)
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Evaluate training accuracy.
with torch.no_grad():
    correct = 0
    for data, label in dataloader:
        output = model(data)
        pred = (output.squeeze() > 0.5).float()
        correct += (pred == label).sum().item()
    accuracy = correct / len(dataset)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
