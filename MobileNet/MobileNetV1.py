import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import numpy as np

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

class DWConv(nn.Module):
    def __init__(self, in_channels, multiplier, stride, padding = 1):
        super(DWConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.multiplier = multiplier
        self.out_channels = self.in_channels * self.multiplier
        self.DW = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      groups=self.in_channels,
                      kernel_size = 3,
                      stride = self.stride,
                      padding = self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels = self.out_channels,
                      out_channels=self.out_channels,
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.DW(x)
        return x

class conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(conv3x3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                stride = stride,
                padding = padding,
                kernel_size=3
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class MobileNetV1(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MobileNetV1, self).__init__()
        self.features = nn.Sequential(
            conv3x3(in_channels, 32, 2, 1),
            DWConv(32, 1, 1),
            conv3x3(32, 64, 1, 1),
            DWConv(64, 1, 2),
            conv3x3(64, 128, 1, 1),
            DWConv(128, 1, 2),
            conv3x3(128, 256, 1, 1),
            DWConv(256, 1, 1),
            conv3x3(256, 256, 1, 1),
            DWConv(256, 1, 2),
            conv3x3(256, 512, 1, 1),
            DWConv(512, 1, 1),
            conv3x3(512, 512, 1, 1),
            DWConv(512, 1, 1),
            conv3x3(512, 512, 1, 1),
            DWConv(512, 1, 1),
            conv3x3(512, 512, 1, 1),
            DWConv(512, 1, 1),
            conv3x3(512, 512, 1, 1),
            DWConv(512, 1, 1),
            conv3x3(512, 512, 1, 1),
            DWConv(512, 1, 2),
            conv3x3(512, 1024, 1, 1),
            DWConv(1024, 1, 2),
            conv3x3(1024, 1024, 1, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.squeeze()
        x = self.fc(x)
        x = self.softmax(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 16

train_set = torchvision.datasets.CIFAR10(root = '../data', train = True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR10(root = '../data', train = False, transform=transform, download=True)

train_loader = DataLoader(train_set, shuffle=True, num_workers=1, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=False, num_workers=1, batch_size=batch_size)

EPOCHS = 20
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device('cuda')

model = MobileNetV1(3, 10)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr = 0.001, momentum=0.9, weight_decay=0.0005)

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