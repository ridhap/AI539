# train.sh

# !/bin/bash

pwd >> logs/lenet_mnist.log 
hostname >> logs/lenet_mnist.log 
date >> logs/lenet_mnist.log 
echo "==============================================================================" >> logs/lenet_mnist.log
# Train LeNet on MNIST dataset
echo "Train LeNet on MNIST dataset" >> logs/lenet_mnist.log
python3 main.py --model=lenet --dataset=mnist --train_transform=True >> logs/lenet_mnist.log
echo "==============================================================================" >> logs/lenet_mnist.log
date >> logs/lenet_mnist.log 


pwd >> logs/resnet_cifar10.log
hostname >> logs/resnet_cifar10.log
date >> logs/resnet_cifar10.log
echo "==============================================================================" >> logs/resnet_cifar10.log
# Train ResNet-18 on CIFAR-10 dataset
echo "Train ResNet-18 on CIFAR-10 dataset" >> logs/resnet_cifar10.log
python3 main.py --model=resnet --dataset=cifar10 --train_transform=True>> logs/resnet_cifar10.log
echo "==============================================================================" >> logs/resnet_cifar10.log
date  >> logs/resnet_cifar10.log
