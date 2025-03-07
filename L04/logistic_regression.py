#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# Generate synthetic binary classification data
X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=1.5)
# Convert labels to 0 and 1 (if not already)
y = (y == 1).astype(np.float32)

# Plot the raw data
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.title("Binary Classification Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("logistic_data.png")
plt.close()
print("Data plot saved to logistic_data.png.")

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


# Define a simple logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)


model = LogisticRegressionModel(X.shape[1])
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 100
losses = []
for epoch in range(n_epochs):
    optimizer.zero_grad()
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Plot the loss curve
plt.figure()
plt.plot(losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("logistic_loss_curve.png")
plt.close()
print("Loss curve saved to logistic_loss_curve.png.")

# Plot decision boundary
# Create a grid of points
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32)
with torch.no_grad():
    probs = model(grid_tensor).reshape(xx.shape).numpy()

plt.figure()
plt.contourf(xx, yy, probs, levels=np.linspace(0, 1, 10), cmap='bwr', alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("logistic_decision_boundary.png")
plt.close()
print("Decision boundary saved to logistic_decision_boundary.png.")
