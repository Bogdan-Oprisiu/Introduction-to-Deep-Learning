from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size=128, download=True):
    """
    Downloads the MNIST dataset and returns DataLoaders for training and testing.

    For GAN training, we normalize the images to the range [-1, 1] so that the generator's
    output (typically using Tanh activation) matches the data distribution.

    Args:
        batch_size (int): Number of images per batch.
        download (bool): Whether to download the dataset if not already present.

    Returns:
        train_loader, test_loader (DataLoader, DataLoader): Data loaders for training and test sets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize MNIST images to [-1, 1]
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=download, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(batch_size=128)
    print("Number of training batches:", len(train_loader))
    print("Number of test batches:", len(test_loader))
