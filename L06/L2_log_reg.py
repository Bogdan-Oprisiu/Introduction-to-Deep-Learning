#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

# Generate synthetic binary classification data.
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, flip_y=0.1, random_state=42)

# Plot the data.
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("Synthetic Binary Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("l2_logreg_data.png")
plt.close()
print("Data plot saved to l2_logreg_data.png.")

# Convert data to torch tensors.
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define logistic regression model.
class LogisticRegressionL2(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionL2, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionL2(input_dim=2)
criterion = nn.BCELoss()
# Use weight_decay in optimizer for L2 regularization.
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)

num_epochs = 200
losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Plot loss curve.
plt.figure()
plt.plot(losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Logistic Regression with L2 Regularization")
plt.savefig("l2_logreg_loss.png")
plt.show()
print("Loss plot saved as l2_logreg_loss.png.")

# Plot decision boundary.
w = model.linear.weight.detach().numpy()[0]
b = model.linear.bias.item()
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
xx = np.linspace(x_min, x_max, 100)
yy = -(b + w[0]*xx) / w[1]

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.plot(xx, yy, 'k--', label='Decision Boundary')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary for Logistic Regression")
plt.legend()
plt.savefig("l2_logreg_boundary.png")
plt.show()
print("Decision boundary plot saved as l2_logreg_boundary.png.")
