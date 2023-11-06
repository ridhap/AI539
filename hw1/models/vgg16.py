import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()

        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=1, padding=1))  # Set stride to 1 and add padding
        self.vgg_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.vgg_block(x)

class VGG16(nn.Module):
    def __init__(self, num_classes=10, dataset='mnist'):
        super(VGG16, self).__init__()
        if dataset.lower() == 'mnist':
            input_channels = 1
        elif dataset.lower() == 'cifar10':
            input_channels = 3
        else:
            raise ValueError("Invalid dataset name. Supported datasets: 'mnist', 'cifar10'")
        
        self.features = nn.Sequential(
            VGGBlock(input_channels, 64, 2),
            VGGBlock(64, 128, 2),
            VGGBlock(128, 256, 3),
            VGGBlock(256, 512, 3),
            VGGBlock(512, 512, 3)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

