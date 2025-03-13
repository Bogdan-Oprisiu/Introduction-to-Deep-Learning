# visualize.py

import os

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from generator import Generator


def visualize_generated_vs_real(generator, device, batch_size=64, latent_dim=100):
    # Generate fake images from random noise
    noise = torch.randn(batch_size, latent_dim, device=device)
    generator.eval()
    with torch.no_grad():
        fake_images = generator(noise)
    # Generator outputs are in [-1,1] due to Tanh, so convert them to [0,1]
    fake_images = (fake_images + 1) / 2.0

    # Load real MNIST test images (they're normalized to [-1,1] in our setup, so convert to [0,1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    real_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    real_images, _ = next(iter(real_loader))
    real_images = (real_images + 1) / 2.0

    # Create image grids for visualization
    grid_real = make_grid(real_images, nrow=8, padding=2)
    grid_fake = make_grid(fake_images, nrow=8, padding=2)

    # Plot the grids side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(grid_real.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Real MNIST Images")
    axes[0].axis("off")
    axes[1].imshow(grid_fake.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Generated Images")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    # Instantiate the generator model
    generator = Generator(latent_dim=latent_dim).to(device)
    checkpoint_path = os.path.join("../checkpoints", "generator.pth")
    if os.path.exists(checkpoint_path):
        generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded generator checkpoint from", checkpoint_path)
    else:
        print("Generator checkpoint not found. Please train the network first.")
        return

    visualize_generated_vs_real(generator, device)


if __name__ == "__main__":
    main()
