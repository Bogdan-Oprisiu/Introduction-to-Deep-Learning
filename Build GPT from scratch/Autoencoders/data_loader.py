import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms


# Custom transform to add Gaussian noise to a tensor.
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def get_dataloaders(batch_size=128, augment_factor=1):
    """
    Returns train and test dataloaders for the MNIST dataset.
    The training data is augmented by creating additional copies with random transformations.

    Parameters:
        batch_size (int): Batch size for training and test dataloaders.
        augment_factor (int): Number of augmented copies to add to the original training dataset.

    Returns:
        train_loader, test_loader (DataLoader, DataLoader): The data loaders for training and testing.
    """
    # Original transform: convert to tensor and normalize.
    original_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Augmented transform: random rotations, affine transformations, then convert to tensor,
    # normalize and finally add Gaussian noise.
    augmented_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Rotate image by up to 10 degrees.
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine transformation.
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(0., 0.1)  # Add Gaussian noise.
    ])

    # Load the original MNIST training dataset.
    train_dataset_original = datasets.MNIST(root='./data', train=True, download=True, transform=original_transform)

    # Create one or more augmented versions of the training dataset.
    augmented_datasets = [
        datasets.MNIST(root='./data', train=True, download=True, transform=augmented_transform)
        for _ in range(augment_factor)
    ]

    # Combine the original and augmented training datasets.
    combined_train_dataset = ConcatDataset([train_dataset_original] + augmented_datasets)

    # Test dataset remains unchanged.
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=original_transform)

    # Enable pin_memory if a CUDA device is available (helps with faster data transfer).
    pin_memory = torch.cuda.is_available()

    # Create DataLoaders.
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return train_loader, test_loader


if __name__ == '__main__':
    # Set augment_factor to the number of augmented copies you want to add.
    train_loader, test_loader = get_dataloaders(batch_size=128, augment_factor=2)
    print("Number of training batches:", len(train_loader))
    print("Number of test batches:", len(test_loader))
