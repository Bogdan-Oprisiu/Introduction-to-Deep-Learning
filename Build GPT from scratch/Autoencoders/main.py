# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, datasets

from models import RegularAutoencoder, VariationalAutoencoder


# Custom transform to add Gaussian noise to a tensor.
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def get_autoencoder_dataloaders(batch_size=128, augment_factor=1):
    """
    Returns train and test DataLoaders for autoencoder training on MNIST.
    This version does NOT normalize the images so that pixel values remain in [0,1].
    Augmentation is applied if augment_factor > 0.
    """
    # For autoencoder training, we want the pixels in [0,1]
    original_transform = transforms.ToTensor()

    # Augmented transform: apply random transformations, convert to tensor, add noise, and then clamp values to [0,1].
    augmented_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Rotate up to 10 degrees.
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.1),
        transforms.Lambda(lambda t: t.clamp(0., 1.))  # Clamp the noisy image to [0, 1].
    ])

    # Load the original MNIST training dataset.
    train_dataset_original = datasets.MNIST(root='./data', train=True, download=True, transform=original_transform)

    # Create augmented copies if needed.
    augmented_datasets = [
        datasets.MNIST(root='./data', train=True, download=True, transform=augmented_transform)
        for _ in range(augment_factor)
    ]

    # Combine original and augmented datasets.
    combined_train_dataset = ConcatDataset([train_dataset_original] + augmented_datasets)

    # Test dataset (no augmentation)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=original_transform)

    # Enable pin_memory if using CUDA.
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return train_loader, test_loader


def train_regular(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    criterion = nn.BCELoss(reduction='sum')

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Regular AE Epoch {epoch} Batch {batch_idx} Loss: {loss.item() / data.size(0):.6f}")

    avg_loss = train_loss / len(train_loader.dataset)
    print(f"====> Regular AE Epoch {epoch} Average Loss: {avg_loss:.4f}")


def validate_regular(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    criterion = nn.BCELoss(reduction='sum')

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            test_loss += criterion(output, data).item()

    avg_loss = test_loss / len(test_loader.dataset)
    print(f"====> Regular AE Test Loss: {avg_loss:.4f}")


def train_vae(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    BCE = nn.BCELoss(reduction='sum')

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = BCE(reconstruction, data)
        # KL divergence loss.
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = bce_loss + kl_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"VAE Epoch {epoch} Batch {batch_idx} Loss: {loss.item() / data.size(0):.6f}")

    avg_loss = train_loss / len(train_loader.dataset)
    print(f"====> VAE Epoch {epoch} Average Loss: {avg_loss:.4f}")


def validate_vae(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    BCE = nn.BCELoss(reduction='sum')

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = BCE(reconstruction, data)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = bce_loss + kl_loss
            test_loss += loss.item()

    avg_loss = test_loss / len(test_loader.dataset)
    print(f"====> VAE Test Loss: {avg_loss:.4f}")


def main():
    print("Welcome to the MNIST Autoencoder Trainer!")
    # Ask which model to train.
    model_type = input("Which model would you like to train? (regular/vae): ").strip().lower()
    while model_type not in ("regular", "vae"):
        model_type = input("Please enter either 'regular' or 'vae': ").strip().lower()

    # Get training hyperparameters.
    try:
        epochs = int(input("Enter number of epochs (e.g., 10): "))
        batch_size = 128
        lr = 0.001
        augment_factor = 2
    except ValueError:
        print("Invalid input. Exiting.")
        return

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the autoencoder-specific dataloader (without normalization).
    train_loader, test_loader = get_autoencoder_dataloaders(batch_size=batch_size, augment_factor=augment_factor)

    # Instantiate the chosen model.
    latent_dim = 20
    if model_type == "regular":
        model = RegularAutoencoder(latent_dim=latent_dim)
    else:
        model = VariationalAutoencoder(latent_dim=latent_dim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop.
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        if model_type == "regular":
            train_regular(model, device, train_loader, optimizer, epoch)
            validate_regular(model, device, test_loader)
        else:
            train_vae(model, device, train_loader, optimizer, epoch)
            validate_vae(model, device, test_loader)

    # Save the model checkpoint.
    checkpoint_file = f"{model_type}_autoencoder.pth"
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Training complete! Model saved to {checkpoint_file}")


if __name__ == "__main__":
    main()
