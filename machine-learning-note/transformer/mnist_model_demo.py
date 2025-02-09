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
parser.add_argument("--model", type=str, default="FashionSimpleNet", choices=["FashionSimpleNet"])
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--nepochs", type=int, default=200, help="max epochs")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FashionMNIST"])
args = parser.parse_args()



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



# model structure
class FashionSimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.classifier(x)
        return x


# Define transforms.
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Create dataloaders. Use pin memory if cuda.
# TODO: how to use test-set
home_path = os.environ["HOME"]
mnist_path = f"{home_path}/workspace/machine-learning/fashion-mnist/data/"
fashion_mnist_path = f"{home_path}/workspace/machine-learning/fashion-mnist/data/"
if args.dataset == "MNIST":
    trainset = datasets.MNIST(mnist_path, train=True, download=True, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers)
    valset = datasets.MNIST(mnist_path, train=False, transform=data_transforms)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers)
    print("Training on MNIST")
    print("trainset shape:", trainset.data.shape)
    print("valset shape:", valset.data.shape)
    print("class:", trainset.classes)
elif args.dataset == "FashionMNIST":
    trainset = datasets.FashionMNIST(fashion_mnist_path, train=True, download=True, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers)
    valset = datasets.FashionMNIST(fashion_mnist_path, train=False, transform=data_transforms)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers)
    print("Training on FashionMNIST")
    print("trainset shape:", trainset.data.shape)
    print("valset shape:", valset.data.shape)
    print("class:", trainset.classes)
else:
    print(f"invalid dataset {args.dataset}")
    sys.exit(1)


'''
Training on MNIST
trainset shape: torch.Size([60000, 28, 28])
valset shape: torch.Size([10000, 28, 28])
class: ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

Training on FashionMNIST
trainset shape: torch.Size([60000, 28, 28])
valset shape: torch.Size([10000, 28, 28])
class: ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
'''


def run_model(net, loader, loss_fn, optimizer, device, train=True):
    # Set mode
    if train:
        net.train()
    else:
        net.eval()
    running_loss = 0
    running_accuracy = 0
    for _, (X, y) in enumerate(loader):
        # Pass to gpu or cpu
        X, y = X.to(device), y.to(device)
        # Zero the gradient
        optimizer.zero_grad()
        # forward calculation
        with torch.set_grad_enabled(train):
            output = net(X)
            _, pred = torch.max(output, 1)
            loss = loss_fn(output, y)
        # If on train backpropagate
        if train:
            loss.backward()
            optimizer.step()
        # Calculate stats
        running_loss += loss.item()
        running_accuracy += torch.sum(pred == y.detach())
    return running_loss / len(loader), running_accuracy.double() / len(loader.dataset)


if __name__ == "__main__":
    # Set up the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on {}".format(device))

    # Set seeds. If using numpy this must be seeded too.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Setup folders for saved models
    if not os.path.exists("saved-models/"):
        os.mkdir("saved-models/")
    # Setup folders. Each run must have it"s own folder. Creates
    # a logs folder for each model and each run.
    if not os.path.exists("logs/"):
        os.mkdir("logs/")
    out_dir = "logs/{}".format(args.model)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    run = 0
    current_dir = "{}/run-{}".format(out_dir, run)
    while os.path.exists(current_dir):
        run += 1
        current_dir = "{}/run-{}".format(out_dir, run)
    os.mkdir(current_dir)
    logfile = open("{}/log.txt".format(current_dir), "w")
    print(args, file=logfile)

    # Init network, loss_fn and early stopping
    #print(model.__dict__)
    """
    FashionSimpleNet <class "model.FashionSimpleNet">
    resnet18 <function resnet18 at 0x15b691580>
    resnet34 <function resnet34 at 0x15b691a80>
    resnet50 <function resnet50 at 0x15b691b20>
    resnet101 <function resnet101 at 0x15b691bc0>
    resnet152 <function resnet152 at 0x15b691c60>
    """
    #net = model.__dict__[args.model]().to(device)
    net = FashionSimpleNet().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    # Train the network
    patience = args.patience
    best_loss = 1e4
    best_accuray = 1.0
    writeFile = open("{}/stats.csv".format(current_dir), "a")
    writer = csv.writer(writeFile)
    writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"])
    for e in range(args.nepochs):
        start = time.time()
        train_loss, train_acc = run_model(net, train_loader, loss_fn, optimizer, device, train=True)
        val_loss, val_acc = run_model(net, val_loader, loss_fn, optimizer, device, train=False)
        end = time.time()
        print(f"Epoch: {e+1}\t train loss: {train_loss:.3f}, train acc: {train_acc:.3f}, val loss: {val_loss:.3f}, val acc: {val_acc:.3f}, time: {end-start:.1f}s")
        # Write to csv file
        writer.writerow([e+1, train_loss, train_acc.item(), val_loss, val_acc.item()])
        # early stopping and save best model
        if val_loss < best_loss:
            best_accuray = val_acc
            best_loss = val_loss
            patience = args.patience
            #model_state = {
            #    "arch": args.model,
            #    "state_dict": net.state_dict()
            #}
            #torch.save(model_state, f"saved-models/{args.model}-{args.dataset}-run-{run}.pth.tar")
            # you can view the model structure in Netron
            torch.save(net, f"saved-models/{args.model}-{args.dataset}-run-{run}.pth")
        else:
            patience -= 1
            if patience == 0:
                print("Run out of patience!")
                writeFile.close()
                break
    print(f"{args.model}.{args.dataset} validation best_loss: {best_loss:.4f}, best_accuray: {best_accuray:.4f}")


'''
FashionSimpleNet.FashionMNIST validation best_loss: 0.2037, best_accuray: 0.9290
FashionSimpleNet.MNIST validation best_loss: 0.0191, best_accuray: 0.9936
'''