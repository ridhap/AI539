import torch
from torchvision import datasets, transforms

def load_mnist(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomRotation(90),  # Randomly rotate the images by 5 degrees
        # transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

def load_cifar10(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomRotation(90),  # Randomly rotate the images by 5 degrees
        # transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader
