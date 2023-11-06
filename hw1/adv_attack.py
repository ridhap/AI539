import torch
import torchvision.transforms as transforms

def PGD(x, y, model, loss, model_name, epsilon, norm_mean=None, norm_std=None, niter=5, stepsize=2/255, randinit=True):
    
    x = x.to(device='cuda')
    y = y.to(device='cuda')
    ori_image = x.data
    
    if randinit:
       x = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
       x = torch.clamp(x, min=0, max=1)

    for i in range(niter) :    
        x.requires_grad = True
        output = model(x)

        model.zero_grad()
        cost = loss(output, y).to(device)
        cost.backward()

        data_grad = x.grad.data
        adv_image = x + stepsize*data_grad.sign()
        eta = adv_image - ori_image
        eta_clip = torch.clamp(eta, min=-epsilon, max=epsilon)
        image = torch.clamp(ori_image + eta_clip, min=0, max=1)
        # image = transforms.Normalize(mean = norm_mean, std = norm_std)(image).detach()
            
    return image


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import torch.backends.cudnn as cudnn
    import torchvision
    import torchvision.transforms as transforms
    import os
    import argparse
    import tqdm
    from models import *
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='lenet', type=str , help='model name')
    parser.add_argument('--dataset', default='mnist', type=str , help='dataset name')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Data
    print('==> Preparing data..')
    if args.dataset=='cifar10':
        transform_test = transforms.Compose([#transforms.RandomCrop(32, padding=4),
                                            #transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    if args.dataset=='mnist':
        transform_test = transforms.Compose([#transforms.Resize(256),transforms.RandomCrop(224),
                                            #transforms.RandomHorizontalFlip(),
                                            # transforms.RandomRotation(180),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])

        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    # Model
    if args.dataset=='cifar10':
       if args.model=='resnet':
          model = ResNet18()
    elif args.dataset=='mnist':
       if args.model=='lenet':
          model = LeNet_mnist()
    model=model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    for model_pth in ['adv_train']:#,'dropout','wd_1e-05', 'wd_0.0001', 'wd_0.001','wd_0.01', 'wd_0.1']:    
        # model_path=f'/home/ridha/Documents/AI539Trustworthy/hw2/trained_models/{args.model}_{args.dataset}_{model_pth}.pth'
        model_path=f'/home/ridha/Documents/AI539Trustworthy/hw2/trained_models/{args.model}_{args.dataset}.pth'

        print(model_path)
        checkpoint=torch.load(model_path)
        model.load_state_dict(checkpoint['net'])

        loss = nn.CrossEntropyLoss()

        if args.model=='resnet':
        #    norm_mean = [0.4914, 0.4822, 0.4465]
        #    norm_std = [0.2023, 0.1994, 0.2010]
           epsilon = 0.03
        elif args.model=='lenet':
        #    norm_mean = [0.1307]
        #    norm_std = [0.3081]
           epsilon = 0.3
        
        
        correct = 0
        for batch_idx, (input, target) in enumerate(tqdm.tqdm(testloader)):
            #print(batch_idx)
            input, target = input.to(device), target.to(device)
            output = model(input)
            init_pred = output.max(1, keepdim=True)[1]
            #correct += init_pred.eq(target).sum().item()
            if init_pred.item() == target.item():
                correct += 1

        final_acc = correct/float(len(testloader))
        print("Correct Samples(clean):", correct,", Test Accuracy(clean): ", final_acc)
        
        
        correct = 0
        for batch_idx, (input, target) in enumerate(tqdm.tqdm(testloader)):
            #print(batch_idx)
            input, target = input.to(device), target.to(device)
            output = model(input)
            init_pred = output.max(1, keepdim=True)[1]
            if init_pred.item() != target.item():
                continue
            # perturbed_data = PGD(input, target, model, loss, args.model, epsilon, norm_mean, norm_std)
            perturbed_data = PGD(input, target, model, loss, args.model, epsilon)

            output = model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                correct += 1
        final_acc = correct/float(len(testloader))
        print("Correct Samples(adv):", correct," Test Accuracy(adv):", final_acc) 

    """
    acc_arr=[]
    iter_arr=[1, 2, 3, 4, 5, 10, 20, 30, 40, 80] 
    for niter in iter_arr:
      correct = 0
      for batch_idx, (input, target) in enumerate(tqdm.tqdm(testloader)):
        #print(batch_idx)
        input, target = input.to(device), target.to(device)
        output = model(input)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        perturbed_data = PGD(input, target, model, loss, args.model, epsilon, norm_mean, norm_std, niter=niter)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
      acc_arr.append(final_acc)
      final_acc = correct/float(len(testloader))
      print("Correct Samples(adv):", correct," Iterations:", niter," Test Accuracy(adv):", final_acc)
    print("Iterations:", iter_arr)
    print("Iteration Accuracy Array:", acc_arr)
    
    plt.plot(iter_arr,acc_arr,'-o')
    plt.title(f"Adv_Attack)Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.savefig(f'./reports/Adv_Attack_Iterations_{args.model}.png')
    plt.clf()
   
    acc_arr=[]
    epsilon_arr=[0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0] 
    for epsilon in epsilon_arr:
      correct = 0
      for batch_idx, (input, target) in enumerate(tqdm.tqdm(testloader)):
        #print(batch_idx)
        input, target = input.to(device), target.to(device)
        output = model(input)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        perturbed_data = PGD(input, target, model, loss, args.model, epsilon, norm_mean, norm_std)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
      acc_arr.append(final_acc)
      final_acc = correct/float(len(testloader))
      print("Correct Samples(adv):", correct," Epsilon:", epsilon," Test Accuracy(adv):", final_acc)
    print("Epsilon:", epsilon_arr)
    print("Epsilon Accuracy Array:", acc_arr)
    
    plt.plot(epsilon_arr,acc_arr,'-o')
    plt.title(f"Adv_Attack_Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.savefig(f'./reports/Adv_Attack_Epsilon_{args.model}.png')
    plt.clf()
    """
    
    
