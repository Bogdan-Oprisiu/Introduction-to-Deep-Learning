import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        """
        Discriminator network for MNIST.
        """
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: (1, 28, 28)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()  # Output probability between 0 and 1.
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    # Quick test of the discriminator.
    disc = Discriminator()
    # Create a batch of 16 random images.
    x = torch.randn(16, 1, 28, 28)
    pred = disc(x)
    print("Discriminator output shape:", pred.shape)  # Expected: (16, 1)
