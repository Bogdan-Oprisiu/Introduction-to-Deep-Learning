import matplotlib.pyplot as plt
import torch

from data_loader import get_dataloaders
from models import RegularAutoencoder, VariationalAutoencoder


def show_reconstructions(model, device, test_loader, num_images=8):
    """
    Takes a model and a test_loader, gets a batch of test images,
    computes reconstructions, and then displays original vs. reconstructed images.
    """
    model.eval()
    # Get one batch of test images.
    images, _ = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        # For VAE, the forward returns (reconstruction, mu, logvar); for regular AE, it returns reconstruction.
        if hasattr(model, 'reparameterize'):
            reconstructions, _, _ = model(images)
        else:
            reconstructions = model(images)

    # Move tensors to CPU and convert to numpy arrays.
    images = images.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()

    # Plot original images on top and reconstructions below.
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        # Original image.
        axes[0, i].imshow(images[i, 0, :, :], cmap='gray')
        axes[0, i].axis('off')
        # Reconstruction.
        axes[1, i].imshow(reconstructions[i, 0, :, :], cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    # Ask user for model selection.
    model_type = input("Which model do you want to visualize? (regular/vae): ").strip().lower()
    while model_type not in ("regular", "vae"):
        model_type = input("Please enter either 'regular' or 'vae': ").strip().lower()

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Instantiate the selected model.
    latent_dim = 20
    if model_type == "regular":
        model = RegularAutoencoder(latent_dim=latent_dim)
        checkpoint_file = "regular_autoencoder.pth"
    else:
        model = VariationalAutoencoder(latent_dim=latent_dim)
        checkpoint_file = "vae_autoencoder.pth"

    # Load the model checkpoint.
    try:
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        model.to(device)
        print(f"Loaded model weights from {checkpoint_file}")
    except Exception as e:
        print("Error loading model checkpoint:", e)
        return

    # Load test data (no augmentation for test).
    _, test_loader = get_dataloaders(batch_size=16, augment_factor=0)

    # Ask user how many images they want to visualize.
    try:
        num_images = int(input("Enter the number of images to visualize (default 8): ") or 8)
    except ValueError:
        num_images = 8

    # Display reconstructions.
    show_reconstructions(model, device, test_loader, num_images=num_images)


if __name__ == "__main__":
    main()
