#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    # Clip predictions to avoid log(0)
    p = np.clip(y_pred, 1e-12, 1. - 1e-12)
    log_likelihood = -np.log(p[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss


def predict(X, W, b):
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    return np.argmax(probs, axis=1)


def main():
    # Generate synthetic data for 3 classes
    X, y = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=1.5)

    # Plot the data
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.title("Softmax Regression Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("softmax_data.png")
    plt.close()
    print("Data plot saved to softmax_data.png.")

    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # Initialize weights and bias
    np.random.seed(42)
    W = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros((1, n_classes))

    # Training hyperparameters
    learning_rate = 0.1
    n_iterations = 200
    losses = []

    # One-hot encode y for gradient computation
    y_onehot = np.eye(n_classes)[y]

    for i in range(n_iterations):
        # Forward pass: compute logits and softmax probabilities
        logits = np.dot(X, W) + b
        probs = softmax(logits)

        # Compute loss
        loss = cross_entropy_loss(y, probs)
        losses.append(loss)

        # Gradient of loss with respect to logits
        grad_logits = (probs - y_onehot) / n_samples

        # Gradients for W and b
        grad_W = np.dot(X.T, grad_logits)
        grad_b = np.sum(grad_logits, axis=0, keepdims=True)

        # Update parameters
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b

        if i % 20 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    # Plot loss curve
    plt.figure()
    plt.plot(losses, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss Curve - Softmax Regression (Scratch)")
    plt.savefig("softmax_loss_curve.png")
    plt.close()
    print("Loss curve saved to softmax_loss_curve.png.")

    # Plot decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(grid, W, b)
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.title("Decision Boundary - Softmax Regression (Scratch)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("softmax_decision_boundary.png")
    plt.close()
    print("Decision boundary saved to softmax_decision_boundary.png.")


if __name__ == "__main__":
    main()
