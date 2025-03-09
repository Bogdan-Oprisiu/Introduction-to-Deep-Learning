#!/usr/bin/env python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img, title=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis("off")

def main():
    # Define data augmentation transforms.
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 training dataset with augmentation.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

    # Get a batch of augmented training images.
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Visualize the augmented images.
    plt.figure(figsize=(8, 8))
    imshow(torchvision.utils.make_grid(images), title="Augmented CIFAR-10 Images")
    plt.savefig("data_augmentation.png")
    plt.show()
    print("Data augmentation visualization saved as data_augmentation.png.")

if __name__ == "__main__":
    main()
