import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import torch.optim as optim
import time

import random
import torch.backends.cudnn as cudnn

seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)

class DenseUnitBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseUnitBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch1 = nn.BatchNorm2d(self.in_features)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=self.in_features,
            out_channels=self.out_features * 4,
            kernel_size=1
                               )
        self.batch2 = nn.BatchNorm2d(self.out_features * 4)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=self.out_features * 4,
            out_channels=self.out_features,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.net = nn.Sequential(
            self.batch1, self.relu1, self.conv1, self.batch2, self.relu2, self.conv2
        )
        self.proj = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1
        )
    def forward(self, x):
        x2 = self.net(x)
        x = self.proj(x)
        x2 = x2.add(x)
        return x2

class DenseBlock(nn.Module):
    def __init__(self, L, k_0 = 16, k = 12):
        super(DenseBlock, self).__init__()
        self.k = k
        self.k_0 = k_0
        net_list = [DenseUnitBlock(self.k_0 + i * self.k, self.k) for i in range(L)]
        self.net_list = nn.ModuleList(net_list)
    def forward(self, x):
        input_list = [x]
        for block in self.net_list:
            for idx, i in enumerate(input_list):
                if idx == 0:
                    t = i
                else:
                    t = torch.cat((t, i), dim = 1)
            out = block(t)
            input_list.append(out)
        return out

class TransitionLayers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayers, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            kernel_size=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, in_channels, k, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.in_channels = in_channels
        self.batch1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.k * 2,
            kernel_size=7,
            stride=2,
            padding=4
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.net1 = nn.Sequential(
            self.batch1, self.relu1, self.conv1, self.pool1
        )

        self.block1 = DenseBlock(L = 6, k_0 = self.k * 2, k = self.k)
        self.trans1 = TransitionLayers(in_channels=self.k, out_channels=self.k)
        self.block2 = DenseBlock(L = 12, k_0 = self.k, k=self.k)
        self.trans2 = TransitionLayers(in_channels=self.k, out_channels=self.k)
        self.block3 = DenseBlock(L = 24, k_0 = self.k, k=self.k)
        self.trans3 = TransitionLayers(in_channels=self.k ,out_channels=self.k)
        self.block4 = DenseBlock(L = 16, k_0 = self.k, k = self.k)
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_features=self.k, out_features=self.num_classes)
        self.soft = nn.Softmax(dim = 1)
        self.net2 = nn.Sequential(
            self.block1,
            self.trans1,
            self.block2,
            self.trans2,
            self.block3,
            self.trans3,
            self.block4,
            self.flat,
            self.fc,
            self.soft
        )

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)

        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root = '../data', train = True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR10(root = '../data', train = False, transform=transform, download=True)

BATCH_SIZE = 16
train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=1)
test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device('cuda')

model = DenseNet(in_channels=3, k = 32, num_classes = 10)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
# optimizer = optim.AdamW(params=model.parameters(), lr=0.001, weight_decay=0.0005)

EPOCHS = 20

def eval_model(model, data):
    model.eval()
    with torch.no_grad():
        t_count = 0
        a_count = 0
        t_loss = 0
        for j, data2 in enumerate(data):
            inputs, labels = data2[0].to(device), data2[1].to(device)
            t_count += len(inputs)
            preds = model(inputs)
            loss = criterion(preds, labels)
            output = torch.argmax(preds, dim=1)
            a_count += sum(output == labels)
            t_loss += loss
    model.train()
    val_loss = t_loss / t_count
    val_acc = a_count / t_count
    return val_loss, val_acc

if __name__ == "__main__":
    for epoch in range(EPOCHS):
        for dset in ['train', 'test']:
            if dset == 'train':
                model.train()
            else:
                model.eval()
            a_count = 0
            t_count = 0
            t_loss = 0
            if dset == 'train':
                start = time.time()
                for i, data in enumerate(train_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    t_count += len(inputs)
                    optimizer.zero_grad()
                    preds = model(inputs)
                    loss = criterion(preds, labels)
                    loss.backward()
                    optimizer.step()

                    output = torch.argmax(preds, dim = 1)
                    a_count += sum(output == labels)
                    t_loss += loss
                    if (i + 1) % 1000 == 0:
                        print(f"loss : {t_loss / t_count:.5f} acc : {a_count / t_count:.3f}")
                time_delta = time.time() - start
                print(f"Final Train Acc : {a_count / t_count:.3f}")
                print(f"Final Train Loss : {t_loss / t_count:.5f}")
                print(f'Train Finished in {time_delta // 60}mins {time_delta % 60} secs')
            else:
                val_loss, val_acc = eval_model(model, test_loader)
                print(f"Val loss : {val_loss:.5f} Val acc : {val_acc:.3f}")