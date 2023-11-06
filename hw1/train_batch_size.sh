# train.sh

#!/bin/bash

#Train LeNet on MNIST dataset
python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model LeNet --dataset MNIST --bs 32

#Train LeNet on MNIST dataset
python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model LeNet --dataset MNIST --bs 128

# Train ResNet-18 on MNIST dataset
python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model ResNet --dataset CIFAR10 --bs 32

# Train ResNet-18 on MNIST dataset
python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model ResNet --dataset CIFAR10 --bs 128