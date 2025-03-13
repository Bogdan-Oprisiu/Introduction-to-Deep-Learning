import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        """
        Generator network for MNIST.

        Args:
            latent_dim (int): Dimension of the input latent vector.
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # Project and reshape the latent vector to a feature map.
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            # Upsample to (64, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Upsample to (1, 28, 28)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output values in [-1,1]
        )

    def forward(self, z):
        return self.net(z)


if __name__ == '__main__':
    # Quick test of the generator.
    latent_dim = 100
    gen = Generator(latent_dim=latent_dim)
    # Create a batch of 16 random latent vectors.
    z = torch.randn(16, latent_dim)
    fake_images = gen(z)
    print("Fake images shape:", fake_images.shape)  # Expected: (16, 1, 28, 28)