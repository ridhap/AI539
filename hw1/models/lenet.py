# models/lenet.py

import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10, dataset = 'mnist' ):  # For MNIST, input_channels=1; for CIFAR-10, input_channels=3
        super(LeNet, self).__init__()
        if dataset.lower() == 'mnist':
            input_channels = 1
        elif dataset.lower() == 'cifar10':
            input_channels = 3
        else:
            raise ValueError("Invalid dataset name. Supported datasets: 'mnist', 'cifar10'")
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        # Calculate the correct number of input features for the first fully connected layer
        if input_channels == 1:  # For MNIST
            self.num_features = 16 * 4 * 4  # 256
        else:  # For CIFAR-10
            self.num_features = 16 * 5 * 5  # 400

        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


