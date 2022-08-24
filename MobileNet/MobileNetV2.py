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

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, multiplier = 6, stride = 2):
        super(Bottleneck, self).__init__()
        self.stride = stride
        mid_ch = in_channels * multiplier
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_ch,
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=mid_ch,
                out_channels=mid_ch,
                stride=stride,
                kernel_size=3,
                padding=1,
                groups=mid_ch,
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=mid_ch,
                out_channels=out_channels,
                stride=1,
                kernel_size=1
            ),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )
        self.relu = nn.ReLU6()

    def forward(self, x):
        x2 = self.net(x)
        if self.stride == 1:
            x = self.identity(x)
            x2 = torch.add(x2, x)
        x2 = self.relu(x2)
        return x2

def make_layer(n, in_channels, out_channels, stride, multiplier):
    b_l = []
    for i in range(n):
        if i == 0:
            b_l.append(Bottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                multiplier=1
            ))
        else:
            b_l.append(
                Bottleneck(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    multiplier=multiplier
                )
            )
    return b_l


class MobileNetV2(nn.Module):
    def __init__(self, in_channels, num_classes, multiplier=6):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=2
        )
        self.relu1 = nn.ReLU6()
        self.batch1 = nn.BatchNorm2d(32)
        self.bottleneck1 = Bottleneck(
            in_channels=32,
            out_channels=16,
            multiplier=1,
            stride=1
        )
        self.bottleneck2 = nn.Sequential(*make_layer(2, 16, 24, 2, multiplier))
        self.bottleneck3 = nn.Sequential(*make_layer(3, 24, 32, 2, multiplier))
        self.bottleneck4 = nn.Sequential(*make_layer(4, 32, 64, 2, multiplier))
        self.bottleneck5 = nn.Sequential(*make_layer(3, 64, 96, 1, multiplier))
        self.bottleneck6 = nn.Sequential(*make_layer(3, 96, 160, 2, multiplier))
        self.bottleneck7 = nn.Sequential(*make_layer(1, 160, 320, 1, multiplier))

        self.conv2 = nn.Conv2d(
            in_channels=320,
            out_channels=1280,
            kernel_size=1,
            stride=1
        )
        self.batch2 = nn.BatchNorm2d(1280)
        self.relu2 = nn.ReLU6()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv3 = nn.Conv2d(
            in_channels=1280,
            out_channels=num_classes,
            kernel_size=1,
            stride=1
        )
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = x.squeeze()
        x = self.soft(x)
        return x

transform = transforms.Compose([
    transforms.Resize(64),
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

model = MobileNetV2(3, 10)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr = 0.001, momentum=0.9, weight_decay=0.0005)
# optimizer = optim.Adam(params=model.parameters(), lr = 0.001, weight_decay=0.0005)

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
                print(f"Epoch : {epoch} Val loss : {val_loss:.5f} Val acc : {val_acc:.3f}")


