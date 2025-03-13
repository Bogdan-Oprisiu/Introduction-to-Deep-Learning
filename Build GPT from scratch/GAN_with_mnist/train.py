import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from discriminator import Discriminator
from generator import Generator
from setup import get_data_loaders

# Hyperparameters
batch_size = 256
epochs = 100
lr = 0.0002
latent_dim = 100
save_interval = 5  # Save generated images every N epochs

# Create folder for saving generated images and model checkpoints
os.makedirs("images", exist_ok=True)
os.makedirs("../checkpoints", exist_ok=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Get training data loader (we only need training data for GAN training)
    train_loader, _ = get_data_loaders(batch_size=batch_size)

    # Instantiate models
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Fixed noise for consistent image generation
    fixed_noise = torch.randn(64, latent_dim, device=device)

    for epoch in range(1, epochs + 1):
        for i, (imgs, _) in enumerate(train_loader):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            discriminator.zero_grad()
            real_imgs = imgs.to(device)
            valid = torch.ones(real_imgs.size(0), 1, device=device)
            fake = torch.zeros(real_imgs.size(0), 1, device=device)

            # Loss for real images
            output_real = discriminator(real_imgs)
            loss_real = criterion(output_real, valid)

            # Generate fake images
            noise = torch.randn(real_imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(noise)
            output_fake = discriminator(gen_imgs.detach())
            loss_fake = criterion(output_fake, fake)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            generator.zero_grad()
            # Generator tries to make discriminator predict "real" for fake images
            output = discriminator(gen_imgs)
            loss_G = criterion(output, valid)
            loss_G.backward()
            optimizer_G.step()

            # Print training status every 100 batches
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

        # Save generated images at the end of each save_interval epoch
        if epoch % save_interval == 0:
            with torch.no_grad():
                fake_imgs = generator(fixed_noise).detach().cpu()
            # Save a grid of generated images (normalized to [0,1])
            save_image(fake_imgs, os.path.join("images", f"epoch_{epoch}.png"), nrow=8, normalize=True)
            print(f"Saved generated images for epoch {epoch}")

    # Save final model checkpoints
    torch.save(generator.state_dict(), os.path.join("../checkpoints", "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join("../checkpoints", "discriminator.pth"))
    print("Training finished. Models saved.")


if __name__ == "__main__":
    train()
