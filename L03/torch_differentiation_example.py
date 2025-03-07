#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Create a tensor of 100 values linearly spaced between -3 and 3.
    # Set requires_grad=True to enable automatic differentiation.
    x = torch.linspace(-3, 3, 100, requires_grad=True)

    # Define a function f(x) = 3x^3 + 2x^2 - x
    f = 3 * x ** 3 + 2 * x ** 2 - x

    # Compute the gradient of f with respect to x.
    # Since f is not a scalar, we pass grad_outputs as a tensor of ones.
    grad_f = torch.autograd.grad(outputs=f, inputs=x, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    # Print a few sample values
    print("Sample x values:", x[:5])
    print("f(x) for those values:", f[:5])
    print("f'(x) for those values:", grad_f[:5])

    # Convert tensors to numpy arrays for plotting
    x_np = x.detach().numpy()
    f_np = f.detach().numpy()
    grad_np = grad_f.detach().numpy()

    # Plot the function and its derivative
    plt.figure(figsize=(8, 6))
    plt.plot(x_np, f_np, label="f(x) = 3x³ + 2x² - x", color='blue')
    plt.plot(x_np, grad_np, label="f'(x)", color='red', linestyle='--')
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.title("Function and its Derivative (PyTorch autograd)")
    plt.legend()
    plt.grid(True)
    plt.savefig("pytorch_autograd_example.png")
    plt.show()
    print("Plot saved to pytorch_autograd_example.png")


if __name__ == "__main__":
    main()
