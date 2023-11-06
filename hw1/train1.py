import torch
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_mnist, load_cifar10
from models.lenet import LeNet
from models.resnet import ResNet18
from models.vgg16 import VGG16
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
# Parse command-line arguments

parser = argparse.ArgumentParser(description='Train different models on various datasets')
parser.add_argument('--model', type=str, choices=['LeNet', 'VGG16', 'ResNet'], default = "ResNet", help='Model type (LeNet, VGG16, ResNet)')
parser.add_argument('--dataset', type=str, choices=['MNIST', 'CIFAR10'], default = "CIFAR10", help='Dataset type (MNIST, CIFAR-10)')

parser.add_argument('--opti', type=str, default = "SGD", help='Dataset type (MNIST, CIFAR-10)')

parser.add_argument('--lr', type=float, default = 0.01, help='Dataset type (MNIST, CIFAR-10)')
parser.add_argument('--bs', type=int, default = 64, help='Dataset type (MNIST, CIFAR-10)')

args = parser.parse_args()

# Choose your model and dataset based on command-line arguments
if args.model == 'LeNet':
    model = LeNet(dataset=args.dataset)
elif args.model == 'VGG16':
    model = VGG16(  dataset=args.dataset)
elif args.model == 'ResNet':
    model = ResNet18( dataset=args.dataset)
if args.dataset == 'MNIST':
    train_loader, test_loader = load_mnist(batch_size=args.bs)
elif args.dataset == 'CIFAR10':
    train_loader, test_loader = load_cifar10(batch_size=args.bs)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your optimizer and loss function
if args.opti == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
elif args.opti == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

criterion = F.cross_entropy

# Training loop
epochs = 5
training_accuracies = []
testing_accuracies = []
training_losses = []
testing_losses = []

for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Compute and record training accuracy and loss every 5 training iterations (epochs)
    with torch.no_grad():
        model.eval()
        train_output = model(data)
        train_loss = criterion(train_output, target)
        _, train_predicted = torch.max(train_output.data, 1)
        train_accuracy = (train_predicted == target).sum().item() / len(target)

        testing_loss, testing_accuracy = 0, 0
        num_batches = len(test_loader)
        for test_data, test_target in test_loader:
            test_data, test_target = test_data.to(device), test_target.to(device)   
            test_output = model(test_data)
            testing_loss += criterion(test_output, test_target).item()
            _, test_predicted = torch.max(test_output.data, 1)
            testing_accuracy += (test_predicted == test_target).sum().item()

        testing_loss /= num_batches  # Corrected: Divide by the number of batches, not the length of the dataset
        testing_accuracy /= len(test_loader.dataset)
        # Record metrics
        training_accuracies.append(train_accuracy)
        testing_accuracies.append(testing_accuracy)
        training_losses.append(train_loss.item())
        testing_losses.append(testing_loss)

        print('Epoch: {} [{}/{} ({:.0f}%)]\tTraining Loss: {:.6f}\tTesting Loss: {:.6f}\tTraining Accuracy: {:.4f}\tTesting Accuracy: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss.item(), testing_loss, train_accuracy, testing_accuracy))


model_filename = f'{args.model}_{args.dataset}_Model.pth'
torch.save(model.state_dict(), f'/home/ridha/Documents/AI539Trustworthy/trained_models/{model_filename}')  # Save the model


# Plot the training and testing accuracies and losses

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(len(training_accuracies)), training_accuracies, label='Training Accuracy')
plt.plot(range(len(testing_accuracies)), testing_accuracies, label='Testing Accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(len(training_losses)), training_losses, label='Training Loss')
plt.plot(range(len(testing_losses)), testing_losses, label='Testing Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'/home/ridha/Documents/AI539Trustworthy/images/Task_2_Learning_rate/{args.model}_{args.dataset}_{args.opti}_{args.bs}__{args.lr}_Model_horit.png') 


