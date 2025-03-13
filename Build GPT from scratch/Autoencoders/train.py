# train.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from models import RegularAutoencoder, VariationalAutoencoder


def train_regular(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    # Use BCE loss since the decoder outputs values between 0 and 1.
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
        # Compute KL divergence loss
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
    parser = argparse.ArgumentParser(description="Train an Autoencoder on MNIST")
    parser.add_argument("--model", type=str, choices=["regular", "vae"], default="regular",
                        help="Which autoencoder to train: 'regular' or 'vae'")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--augment-factor", type=int, default=0,
                        help="Number of augmented copies to add for data augmentation (default: 0)")
    args = parser.parse_args()

    # Set device to CUDA if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Get the data loaders.
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, augment_factor=args.augment_factor)

    # Instantiate the chosen model.
    if args.model == "regular":
        model = RegularAutoencoder(latent_dim=20)
    else:
        model = VariationalAutoencoder(latent_dim=20)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop.
    for epoch in range(1, args.epochs + 1):
        if args.model == "regular":
            train_regular(model, device, train_loader, optimizer, epoch)
            validate_regular(model, device, test_loader)
        else:
            train_vae(model, device, train_loader, optimizer, epoch)
            validate_vae(model, device, test_loader)

    # Save the trained model.
    checkpoint_file = f"{args.model}_autoencoder.pth"
    torch.save(model.state_dict(), checkpoint_file)
    print("Saved model checkpoint to", checkpoint_file)


if __name__ == "__main__":
    main()
