#!/usr/bin/env python
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def adaline_loss(params, X, y):
    """
    Compute the mean squared error loss for the Adaline model.
    params: a vector containing the weights (first n_features elements) and bias (last element).
    X: input features, shape (n_samples, n_features)
    y: target values, shape (n_samples)
    """
    # Split parameters into weights and bias
    w = params[:-1]
    b = params[-1]
    # Linear prediction
    predictions = np.dot(X, w) + b
    # Mean squared error loss
    loss = np.mean((y - predictions) ** 2)
    return loss


def main():
    # -------------------------------
    # Data Generation and Visualization
    # -------------------------------
    # Generate a synthetic binary classification dataset
    X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=1.5)
    # For Adaline, we transform labels to -1 and 1
    y = np.where(y == 0, -1, 1)

    # Plot and save the dataset
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title("Synthetic Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("data.png")
    print("Data plot saved to data.png.")

    # -------------------------------
    # Adaline Model Training with Autograd
    # -------------------------------
    n_samples, n_features = X.shape

    # Initialize weights and bias (small random numbers)
    params = np.random.randn(n_features + 1) * 0.01

    # Training hyperparameters
    learning_rate = 0.01
    n_iterations = 100
    losses = []

    # Get gradient function for the loss
    loss_grad = grad(adaline_loss)

    # Training loop
    for i in range(n_iterations):
        # Compute current loss
        loss_val = adaline_loss(params, X, y)
        losses.append(loss_val)
        # Compute gradients
        grads = loss_grad(params, X, y)
        # Update parameters using gradient descent
        params = params - learning_rate * grads
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss_val:.4f}")

    # -------------------------------
    # Plot the Loss Curve
    # -------------------------------
    plt.figure()
    plt.plot(losses, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    plt.title("Loss Curve")
    plt.savefig("loss_curve.png")
    print("Loss curve saved to loss_curve.png.")

    # -------------------------------
    # Plot the Decision Boundary
    # -------------------------------
    # For a 2D dataset, decision boundary: w1*x1 + w2*x2 + b = 0
    w = params[:-1]
    b = params[-1]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_vals = np.linspace(x_min, x_max, 100)
    # Solve for x2: x2 = (-b - w1*x1) / w2
    y_vals = (-b - w[0] * x_vals) / w[1]

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.plot(x_vals, y_vals, 'k--', linewidth=2)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.savefig("decision_boundary.png")
    print("Decision boundary plot saved to decision_boundary.png.")

    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
