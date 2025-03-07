#!/usr/bin/env python
import torch
import torch.nn as nn


def main():
    # Suppose we have 3 classes and a batch of 4 examples.
    # Logits: shape (batch_size, num_classes)
    logits = torch.tensor([[2.0, 1.0, 0.1],
                           [0.5, 2.5, 0.3],
                           [1.2, 0.7, 2.0],
                           [0.1, 0.2, 0.3]], requires_grad=True)

    # Targets: class indices (0, 1, or 2) for each example
    targets = torch.tensor([0, 1, 2, 1])

    # Define the cross-entropy loss
    loss_fn = nn.CrossEntropyLoss()

    # Compute loss
    loss = loss_fn(logits, targets)
    print(f"Cross-Entropy Loss: {loss.item():.4f}")

    # Backward pass to compute gradients
    loss.backward()
    print("Gradients on logits:")
    print(logits.grad)


if __name__ == "__main__":
    main()
