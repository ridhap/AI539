import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import sys
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

torch.cuda.synchronize()

import os
import argparse
import tqdm
from models import *
#from utils import progress_bar
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--nepoch', default=50, type=int, help='no.of epochs')
parser.add_argument('--model', default='resnet', type=str , help='model name')
parser.add_argument('--dataset', default='cifar10', type=str , help='dataset name')
parser.add_argument('--train_transform', default=False, type=bool , help='apply tranformation')
parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay in optimizer')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

sys.stdout = open(f'{args.model}_{args.dataset}_{args.weight_decay}.log', 'w')

print("="*120)

print("Model:",args.model)
print("Dataset:",args.dataset)
print("Learning Rate:",args.lr)
print("No.of Epochs:",args.nepoch)
print("Weight Decay:",args.weight_decay)
print("DropOut")
print("="*120)

# Data
print('==> Preparing data..')
if args.dataset=='cifar10':
   if args.train_transform==True:
      transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            #transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
      transform_test = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
   else:
      transform_train = transforms.ToTensor()
      transform_test = transforms.ToTensor()

   trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)

   testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
   testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=1)

elif args.dataset=='mnist':
   if args.train_transform==True:
      transform_train = transforms.Compose([#transforms.RandomCrop(32, padding=4),#transforms.RandomHorizontalFlip(),
                                            #transforms.Resize(256),transforms.RandomCrop(224),
                                            transforms.RandomRotation(180),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,),(0.3081,))])
      transform_test = transforms.Compose([#transforms.Resize(256),transforms.RandomCrop(224),
                                           #transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(180),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))])
   else:
      transform_train = transforms.ToTensor()
      transform_test = transforms.ToTensor()
   
   trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)

   testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
   testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=1)
else:
  print("No such dataset")


print("="*120)


# Model
print('==> Building model..')
if args.dataset=='cifar10':
   if args.model=='resnet':
      net = ResNet18()
elif args.dataset=='mnist':
   if args.model=='lenet':
      net = LeNet_mnist()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./wt_decay_5_resnet_models/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


train_loss_arr=[]
test_loss_arr=[]
train_acc_arr=[]
test_acc_arr=[]
accum_steps=2
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss = loss / accum_steps
        loss.backward()
        train_loss += loss.item()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if accum_steps > 1 and (batch_idx + 1) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Train:- Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total)) 
    train_loss_arr.append(train_loss/(batch_idx+1))
    train_acc_arr.append(100.*correct/total)    

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Test:-  Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_loss_arr.append(test_loss/(batch_idx+1))
        test_acc_arr.append(100.*correct/total)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'/home/ridha/Documents/AI539Trustworthy/hw2/wt_decay_5_resnet_models/{args.model}_{args.dataset}_{args.weight_decay}.pth')
        best_acc = acc



for epoch in range(start_epoch, start_epoch+args.nepoch):
    train(epoch)
    test(epoch)
    scheduler.step()

print("="*120)


print(train_loss_arr, test_loss_arr, train_acc_arr, test_acc_arr)

if not os.path.isdir('reports'):
   os.mkdir('reports')

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig.tight_layout(pad=3.0)
ax1,ax2=axs[0],axs[1]
ax1.plot(range(args.nepoch),train_loss_arr)
ax1.plot(range(args.nepoch),test_loss_arr)
ax1.set_title("Model Loss")
ax1.legend(['train','test'])
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.plot(range(args.nepoch),train_acc_arr)
ax2.plot(range(args.nepoch),test_acc_arr)
ax2.set_title("Model Accuracy")
ax2.legend(['train','test'])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy(%)')
plt.savefig(f'./images/resnet_wt_decay/{args.model}_{args.dataset}_{args.weight_decay}.png')


sys.stdout.close()



