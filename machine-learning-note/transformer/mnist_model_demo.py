#!/usr/bin/env python3
#coding=utf-8
import time
import os
import sys
import csv
# from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='FashionSimpleNet', help="model")
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--nepochs", type=int, default=200, help="max epochs")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--data", type=str, default='MNIST', help="MNIST, or FashionMNIST")
args = parser.parse_args()



class FashionSimpleNet(nn.Module):

    """ Simple network"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, padding=1), # 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.classifier(x)
        return x





#viz
# tsboard = SummaryWriter()

# Set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))

# Set seeds. If using numpy this must be seeded too.
torch.manual_seed(args.seed)
if device== 'cuda:0':
    torch.cuda.manual_seed(args.seed)

# Setup folders for saved models and logs
if not os.path.exists('saved-models/'):
    os.mkdir('saved-models/')
if not os.path.exists('logs/'):
    os.mkdir('logs/')

# Setup folders. Each run must have it's own folder. Creates
# a logs folder for each model and each run.
out_dir = 'logs/{}'.format(args.model)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
run = 0
current_dir = '{}/run-{}'.format(out_dir, run)
while os.path.exists(current_dir):
    run += 1
    current_dir = '{}/run-{}'.format(out_dir, run)
os.mkdir(current_dir)
logfile = open('{}/log.txt'.format(current_dir), 'w')
print(args, file=logfile)



# Define transforms.
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Create dataloaders. Use pin memory if cuda.
home_path = os.environ["HOME"]
fashion_mnist_path = f"{home_path}/workspace/machine-learning/fashion-mnist/data/"
mnist_path = f"{home_path}/workspace/fashion-mnist/data/"
if args.data == 'FashionMNIST':
    trainset = datasets.FashionMNIST(fashion_mnist_path, train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.nworkers)
    valset = datasets.FashionMNIST(fashion_mnist_path, train=False, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.nworkers)
    print('Training on FashionMNIST')
else:
    trainset = datasets.MNIST(mnist_path, train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.nworkers)
    valset = datasets.MNIST(mnist_path, train=False, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.nworkers)
    print('Training on MNIST')


def run_model(net, loader, criterion, optimizer, train = True):
    running_loss = 0
    running_accuracy = 0

    # Set mode
    if train:
        net.train()
    else:
        net.eval()


    for i, (X, y) in enumerate(loader):
        # Pass to gpu or cpu
        X, y = X.to(device), y.to(device)

        # Zero the gradient
        optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            output = net(X)
            _, pred = torch.max(output, 1)
            loss = criterion(output, y)

        # If on train backpropagate
        if train:
            loss.backward()
            optimizer.step()

        # Calculate stats
        running_loss += loss.item()
        running_accuracy += torch.sum(pred == y.detach())
    return running_loss / len(loader), running_accuracy.double() / len(loader.dataset)



if __name__ == '__main__':
    # Init network, criterion and early stopping
    #print(model.__dict__)
    """
    FashionSimpleNet <class 'model.FashionSimpleNet'>
    resnet18 <function resnet18 at 0x15b691580>
    resnet34 <function resnet34 at 0x15b691a80>
    resnet50 <function resnet50 at 0x15b691b20>
    resnet101 <function resnet101 at 0x15b691bc0>
    resnet152 <function resnet152 at 0x15b691c60>
    """

    net = model.__dict__[args.model]().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    # Train the network
    patience = args.patience
    best_loss = 1e4
    writeFile = open('{}/stats.csv'.format(current_dir), 'a')
    writer = csv.writer(writeFile)
    writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'])
    for e in range(args.nepochs):
        start = time.time()
        train_loss, train_acc = run_model(net, train_loader,
                                      criterion, optimizer)
        val_loss, val_acc = run_model(net, val_loader,
                                      criterion, optimizer, False)
        end = time.time()

        # print stats
        stats = """Epoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                val loss: {:.3f}, val acc: {:.3f}\t
                time: {:.1f}s""".format(e+1, train_loss, train_acc, val_loss,
                                        val_acc, end - start)
        print(stats)

        # viz
        # tsboard.add_scalar('data/train-loss',train_loss,e)
        # tsboard.add_scalar('data/val-loss',val_loss,e)
        # tsboard.add_scalar('data/val-accuracy',val_acc.item(),e)
        # tsboard.add_scalar('data/train-accuracy',train_acc.item(),e)

        # Write to csv file
        writer.writerow([e+1, train_loss, train_acc.item(), val_loss, val_acc.item()])
        # early stopping and save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience = args.patience
            model_state = {
                'arch': args.model,
                'state_dict': net.state_dict()
            }
            torch.save(model_state, 'saved-models/{}-run-{}.pth.tar'.format(args.model, run))
        else:
            patience -= 1
            if patience == 0:
                print('Run out of patience!')
                writeFile.close()
                # tsboard.close()
                break
