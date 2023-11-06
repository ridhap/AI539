
# This script is used to run the adversarial attack on the models
# The results are stored in the logs folder

pwd >> logs/lenet_mnist_attack_Task2_Iter.log 
hostname >> logs/lenet_mnist_attack_Task2_Iter.log 
date >> logs/lenet_mnist_attack_Task2_Iter.log 
echo "==============================================================================" >> logs/lenet_mnist_attack_Task2_Iter.log
# Train LeNet on MNIST dataset
python3 adv_attack.py --model=lenet --dataset=mnist >> logs/lenet_mnist_attack_Task2_Iter.log
echo "==============================================================================" >> logs/lenet_mnist_attack_Task2_Iter.log
date >> logs/lenet_mnist_attack_Task2_Iter.log

pwd >> logs/resnet_cifar10_attack_Task2_Iter.log
hostname >> logs/resnet_cifar10_attack_Task2_Iter.log
date >> logs/resnet_cifar10_attack_Task2_Iter.log
echo "==============================================================================" >> logs/resnet_cifar10_attack_Task2_Iter.log
# Train ResNet-18 on CIFAR-10 dataset
python3 adv_attack.py --model=resnet --dataset=cifar10 >> logs/resnet_cifar10_attack_Task2_Iter.log
echo "==============================================================================" >> logs/resnet_cifar10_attack_Task2_Iter.log
date >> logs/resnet_cifar10_attack_Task2_Iter.log