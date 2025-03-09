#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim

# Define XOR dataset.
data = torch.tensor([[0., 0.],
                     [0., 1.],
                     [1., 0.],
                     [1., 1.]], dtype=torch.float32)
targets = torch.tensor([[0.],
                        [1.],
                        [1.],
                        [0.]], dtype=torch.float32)


# Define a simple MLP for XOR.
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)
        self.activation = nn.Tanh()  # You can also try ReLU

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


model = XORNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop.
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluate the trained model.
with torch.no_grad():
    output = model(data)
    predictions = (output > 0.5).float()
    print("Final Predictions:")
    print(predictions)
