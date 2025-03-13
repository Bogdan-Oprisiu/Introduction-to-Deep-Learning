# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


########################################
# Regular Autoencoder
########################################

class RegularAutoencoder(nn.Module):
    def __init__(self, latent_dim=20):
        """
        A convolutional autoencoder for MNIST.

        Args:
            latent_dim (int): Dimension of the latent vector.
        """
        super(RegularAutoencoder, self).__init__()
        # Encoder: input shape (batch, 1, 28, 28)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (batch, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (batch, 64, 7, 7)
            nn.ReLU()
        )
        # Fully connected layer to get to latent_dim
        self.fc_enc = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder: map latent vector back to (64, 7, 7)
        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> (batch, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> (batch, 1, 28, 28)
            nn.Sigmoid()  # Since MNIST pixels are normalized between 0 and 1.
        )

    def forward(self, x):
        batch_size = x.size(0)
        # Encode
        x_enc = self.encoder_conv(x)  # (batch, 64, 7, 7)
        x_enc_flat = x_enc.view(batch_size, -1)  # (batch, 64*7*7)
        latent = self.fc_enc(x_enc_flat)  # (batch, latent_dim)
        # Decode
        x_dec_flat = self.fc_dec(latent)  # (batch, 64*7*7)
        x_dec = x_dec_flat.view(batch_size, 64, 7, 7)  # (batch, 64, 7, 7)
        reconstruction = self.decoder_conv(x_dec)  # (batch, 1, 28, 28)
        return reconstruction


########################################
# Variational Autoencoder (VAE)
########################################

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=20):
        """
        A convolutional variational autoencoder for MNIST.

        Args:
            latent_dim (int): Dimension of the latent space.
        """
        super(VariationalAutoencoder, self).__init__()
        # Encoder: same convolutional layers as regular AE.
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (batch, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (batch, 64, 7, 7)
            nn.ReLU()
        )
        # Fully connected layers to produce the mean and log variance.
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder: map latent vector back to image.
        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> (batch, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> (batch, 1, 28, 28)
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        # Encode input image into latent distribution parameters.
        x_enc = self.encoder_conv(x)  # (batch, 64, 7, 7)
        x_enc_flat = x_enc.view(batch_size, -1)  # (batch, 64*7*7)
        mu = self.fc_mu(x_enc_flat)  # (batch, latent_dim)
        logvar = self.fc_logvar(x_enc_flat)  # (batch, latent_dim)
        # Sample z via reparameterization.
        z = self.reparameterize(mu, logvar)  # (batch, latent_dim)
        # Decode the latent variable.
        x_dec_flat = self.fc_dec(z)  # (batch, 64*7*7)
        x_dec = x_dec_flat.view(batch_size, 64, 7, 7)  # (batch, 64, 7, 7)
        reconstruction = self.decoder_conv(x_dec)  # (batch, 1, 28, 28)
        return reconstruction, mu, logvar


########################################
# Test the Models
########################################

if __name__ == "__main__":
    # Create dummy input: a batch of 16 images, 1x28x28.
    dummy_input = torch.randn(16, 1, 28, 28)

    # Test Regular Autoencoder
    reg_ae = RegularAutoencoder(latent_dim=20)
    recon = reg_ae(dummy_input)
    print("Regular AE reconstruction shape:", recon.shape)  # Expected: (16, 1, 28, 28)

    # Test Variational Autoencoder
    vae = VariationalAutoencoder(latent_dim=20)
    recon, mu, logvar = vae(dummy_input)
    print("VAE reconstruction shape:", recon.shape)  # Expected: (16, 1, 28, 28)
    print("Latent mean shape:", mu.shape)  # Expected: (16, 20)
    print("Latent logvar shape:", logvar.shape)  # Expected: (16, 20)
