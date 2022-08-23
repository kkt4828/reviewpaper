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
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                groups=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x2 = self.net(x)
        x = torch.add(x, x2)
        x = self.relu(x)
        return x

class GConv(nn.Module):
    def __init__(self, groups, in_channels, out_channels, channel_shuffle = True, s2 = False):
        super(GConv, self).__init__()
        self.s2 = s2
        self.channel_shuffle = channel_shuffle
        self.groups = groups
        self.out_channels = out_channels
        self.gconv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            groups=self.groups,
            kernel_size=1,
            stride=1,
            padding=0
        )
        if s2:
            self.gconv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
            self.channel_shuffle = False
        self.batch = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.gconv(x)
        x = self.batch(x)
        if self.channel_shuffle:
            x_shape = x.shape
            x = self.relu(x)
            x = x.reshape(x_shape[0], self.groups, self.out_channels//self.groups, x_shape[-2], -1)
            x = torch.transpose(x, 1, 2)
            x = torch.flatten(x, start_dim=1, end_dim=2)
        if self.s2:
            x = self.relu(x)
        return x

class DWConv(nn.Module):
    def __init__(self, in_channels, stride, multiplier = 1):
        super(DWConv, self).__init__()
        out_channels = in_channels * multiplier
        self.net = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=in_channels,
            kernel_size=3,
            padding=1,
            stride=stride
        )
        self.batch = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.net(x)
        x = self.batch(x)
        return x

class ShuffleNetUnit(nn.Module):
    def __init__(self, stride, in_channels, out_channels, groups, s2=False):
        super(ShuffleNetUnit, self).__init__()
        self.s2 = s2
        self.stride = stride
        if stride == 1:
            d = 1
        else:
            d = 2
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        out_channels = out_channels // d
        self.gconv1 = GConv(groups=groups, in_channels=in_channels, out_channels=out_channels, s2=s2)
        self.DWConv = DWConv(in_channels=out_channels, stride = stride)
        self.gconv2 = GConv(groups=groups, in_channels=out_channels, out_channels=out_channels, channel_shuffle=False)
        if s2:
            self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x2 = self.gconv1(x)
        x2 = self.DWConv(x2)
        x2 = self.gconv2(x2)
        if self.stride == 2:
            x = self.avgpool(x)
            if self.s2:
                x = self.identity(x)
            x = torch.concat((x, x2), dim=1)
        else:
            x = torch.add(x, x2)
        x = self.relu(x)
        return x

class ShuffleNetV1(nn.Module):
    def __init__(self, in_channels, groups, num_classes):
        super(ShuffleNetV1, self).__init__()
        self.channels_dict = {
            1 : 144,
            2 : 200,
            3 : 240,
            4 : 272,
            8 : 384
        }
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=24,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        channel_var = self.channels_dict[groups]
        stage2_list = [
            ShuffleNetUnit(
                stride=1,
                in_channels=channel_var,
                out_channels=channel_var,
                groups=groups
            ) for _ in range(3)
        ]
        s2 = [
                 ShuffleNetUnit(
                     stride=2,
                     in_channels=24,
                     out_channels=channel_var,
                     groups=groups,
                     s2=True

            )] + stage2_list
        self.stage2 = nn.Sequential(*s2)
        stage3_list = [
            ShuffleNetUnit(
                stride=1,
                in_channels=channel_var * 2,
                out_channels=channel_var * 2,
                groups=groups
            ) for _ in range(7)
        ]
        s3 = [
            ShuffleNetUnit(
                stride=2,
                in_channels=channel_var,
                out_channels=channel_var * 2,
                groups=groups
            )
        ] + stage3_list
        self.stage3 = nn.Sequential(*s3)
        stage4_list = [
            ShuffleNetUnit(
                stride=1,
                in_channels=channel_var * 4,
                out_channels=channel_var * 4,
                groups=groups
            ) for _ in range(3)
        ]
        s4 = [
            ShuffleNetUnit(
                stride=2,
                in_channels=channel_var * 2,
                out_channels=channel_var * 4,
                groups=groups
            )
        ] + stage4_list
        self.stage4 = nn.Sequential(*s4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=channel_var*4, out_features=num_classes, bias=True)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.squeeze()
        x = self.fc(x)
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

model = ShuffleNetV1(3, 3, 10)
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