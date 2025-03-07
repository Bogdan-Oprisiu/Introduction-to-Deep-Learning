#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Define transformation for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load MNIST training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


# Define softmax regression model (a single linear layer)
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        logits = self.linear(x)
        return logits  # CrossEntropyLoss applies softmax internally


input_dim = 28 * 28
output_dim = 10
model = SoftmaxRegression(input_dim, output_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
n_epochs = 5
for epoch in range(1, n_epochs + 1):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation on a test set
model.eval()
correct = 0
total = 0
misclassified = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        # Save misclassified examples for plotting (only a few)
        for i in range(data.size(0)):
            if len(misclassified) < 10 and predicted[i] != target[i]:
                misclassified.append((data[i].cpu(), target[i].cpu(), predicted[i].cpu()))

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Plot some misclassified images
if misclassified:
    plt.figure(figsize=(10, 4))
    for idx, (img, true_label, pred_label) in enumerate(misclassified):
        plt.subplot(2, 5, idx + 1)
        plt.imshow(img.view(28, 28), cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("mnist_misclassified.png")
    plt.show()
    print("Misclassified images saved to mnist_misclassified.png.")

# Save the trained model
torch.save(model.state_dict(), "softmax_mnist.pth")
print("Model saved to softmax_mnist.pth.")
