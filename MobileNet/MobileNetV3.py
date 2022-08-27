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

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SqueezeExcitation, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            kernel_size=1,
            stride=1
        )
        self.batch1 = nn.BatchNorm2d(out_channels // 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(
            in_channels=out_channels // 4,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )
        self.batch2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch2(x)
        x = self.sigmoid(x)
        return x

class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, mid_ch, stride = 2, se=False, size=3, mode = 'RE'):
        super(bottleneck, self).__init__()
        padding = 1
        if size == 3:
            padding = 1
        elif size == 5:
            padding = 2
        self.se = se
        self.stride = stride
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = mid_ch,
            kernel_size=1,
            stride=1
        )
        self.batch1 = nn.BatchNorm2d(mid_ch)
        if mode == 'RE':
            self.act1, self.act2, self.act3 = nn.ReLU6(), nn.ReLU6(), nn.ReLU6()
        elif mode == 'HS':
            self.act1, self.act2, self.act3 = nn.Hardswish(), nn.Hardswish(), nn.Hardswish()

        self.DWConv = nn.Conv2d(
            in_channels=mid_ch,
            out_channels=mid_ch,
            groups=mid_ch,
            kernel_size=size,
            stride=stride,
            padding=padding
        )
        self.batch2 = nn.BatchNorm2d(mid_ch)

        self.seblock = SqueezeExcitation(
            in_channels=mid_ch,
            out_channels=mid_ch
        )

        self.conv2 = nn.Conv2d(
            in_channels=mid_ch,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )
        self.batch3 = nn.BatchNorm2d(out_channels)
        if self.stride == 1:
            self.identity = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=self.stride
            )

    def forward(self, x):
        x2 = self.conv1(x)
        x2 = self.batch1(x2)
        x2 = self.act1(x2)
        x2 = self.DWConv(x2)
        x2 = self.batch2(x2)
        x2 = self.act2(x2)
        if self.se:
            x3 = self.seblock(x2)
            x2 = torch.mul(x2, x3)
        x2 = self.conv2(x2)
        x2 = self.batch3(x2)
        if self.stride == 1:
            x = self.identity(x)
            x2 = torch.add(x, x2)

        x2 = self.act3(x2)
        return x2

class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )
        self.bneck1 = bottleneck(in_channels=16, out_channels=16, mid_ch=16, stride=1, mode='RE')
        self.bneck2 = nn.Sequential(
            bottleneck(in_channels=16, out_channels=24, mid_ch=64, stride=2, mode='RE'),
            bottleneck(in_channels=24, out_channels=24, mid_ch=72, stride=1, mode='RE')
        )
        self.bneck3 = nn.Sequential(
            bottleneck(in_channels=24, out_channels=40, mid_ch=72, stride=2, mode='RE', size=5, se=True),
            bottleneck(in_channels=40, out_channels=40, mid_ch=120, stride=1, mode='RE', size=5, se=True),
            bottleneck(in_channels=40, out_channels=40, mid_ch=120, stride=1, mode='RE', size=5, se=True)
        )
        self.bneck4 = nn.Sequential(
            bottleneck(in_channels=40, out_channels=80, mid_ch=240, stride=2, mode='HS'),
            bottleneck(in_channels=80, out_channels=80, mid_ch=200, stride=1, mode='HS'),
            bottleneck(in_channels=80, out_channels=80, mid_ch=184, stride=1, mode='HS'),
            bottleneck(in_channels=80, out_channels=80, mid_ch=184, stride=1, mode='HS')
        )
        self.bneck5 = nn.Sequential(
            bottleneck(in_channels=80, out_channels=112, mid_ch=480, stride=1, mode='HS', se=True),
            bottleneck(in_channels=112, out_channels=112, mid_ch=672, stride=1, mode='HS', se=True)
        )
        self.bneck6 = nn.Sequential(
            bottleneck(in_channels=112, out_channels=160, mid_ch=672, stride=2, mode='HS', se=True, size=5),
            bottleneck(in_channels=160, out_channels=160, mid_ch=960, stride=1, mode='HS', se=True, size=5),
            bottleneck(in_channels=160, out_channels=160, mid_ch=960, stride=1, mode='HS', se=True, size=5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=160,
                out_channels=960,
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm2d(960),
            nn.Hardswish()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, stride=1),
            nn.Hardswish()
        )
        self.fc = nn.Conv2d(in_channels=1280, out_channels=num_classes, kernel_size=1, stride=1)
        self.soft = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bneck1(x)
        x = self.bneck2(x)
        x = self.bneck3(x)
        x = self.bneck4(x)
        x = self.bneck5(x)
        x = self.bneck6(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.fc(x)
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

model = MobileNetV3_Large(10, 3)
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


