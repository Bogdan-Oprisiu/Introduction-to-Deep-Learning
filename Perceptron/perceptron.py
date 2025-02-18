import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Load dataset from file
file_path = "perceptron_toydata.txt"
data = np.loadtxt(file_path)

# Extract features (X) and labels (y)
X = data[:, :2]  # First two columns are features
y = data[:, 2]   # Last column is the label

# Convert to PyTorch tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define a simple Perceptron model with Step Function
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(2, 1, bias=True)  # Two input features, one output neuron

    def forward(self, x):
        # Apply the step function using torch.heaviside.
        # Note: The second argument to torch.heaviside is the value at 0.
        return torch.heaviside(self.fc(x), torch.tensor(0.0))

# Initialize model
model = Perceptron()

# Perceptron Learning Rule Optimizer
def perceptron_learning_rule(model, X_train, y_train, epochs=1000):
    for epoch in range(epochs):
        total_errors = 0
        for i in range(X_train.shape[0]):
            xi = X_train[i].view(1, -1)  # Reshape as row vector (1,2)
            yi = y_train[i]

            output = model(xi)  # Input as row vector (1,2)
            error = (yi - output).squeeze().item()  # Convert to scalar

            if error != 0:  # Update if there's an error
                model.fc.weight.data += error * xi  # Weight update (xi is (1,2))
                model.fc.bias.data += error  # Bias update
                total_errors += 1

        if total_errors == 0:  # Convergence check
            print(f"Training converged at epoch {epoch + 1}")
            break

# Train the perceptron
perceptron_learning_rule(model, X_train, y_train)

# Extract model parameters for decision boundary
w = model.fc.weight.data.numpy()  # Shape is (1,2)
b = model.fc.bias.data.numpy()      # Shape is (1,)

# Compute decision boundary: w[0,0] * x + w[0,1] * y + b[0] = 0  =>  y = -(w[0,0]*x + b[0]) / w[0,1]
x_boundary = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
y_boundary = -(w[0, 0] * x_boundary + b[0]) / w[0, 1]  # Correct indexing

# Plot decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.8)
plt.plot(x_boundary, y_boundary, 'k', linewidth=2)  # Decision boundary
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Boundary (Perfect Classification)')
plt.show()

# Calculate Accuracy
with torch.no_grad():
    predictions = model(X_train)
    accuracy = (predictions == y_train).float().mean().item()
    print(f'Final Training Accuracy: {accuracy * 100:.2f}%')
