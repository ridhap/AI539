# train.sh

#!/bin/bash

# Train LeNet on MNIST dataset
python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model LeNet --dataset MNIST

# # Train LeNet on CIFAR-10 dataset
python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model LeNet --dataset CIFAR10

# # Train ResNet-18 on MNIST dataset
python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model ResNet --dataset MNIST

# Train ResNet-18 on CIFAR-10 dataset
python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model ResNet --dataset CIFAR10

# Train VGG-16 on MNIST dataset
python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model VGG16 --dataset MNIST

# Train VGG-16 on CIFAR-10 dataset
python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model VGG16 --dataset CIFAR10
