# train.sh

#!/bin/bash

#Train LeNet on MNIST dataset
python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model LeNet --dataset MNIST

#Train LeNet on MNIST dataset
# python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model LeNet --dataset MNIST 

# Train ResNet-18 on MNIST dataset
python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model ResNet --dataset CIFAR10 

# Train ResNet-18 on MNIST dataset
# python3 /home/ridha/Documents/AI539Trustworthy/train1.py --model ResNet --dataset CIFAR10 --lr 0.0001