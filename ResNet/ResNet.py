import torch
import torch.nn as nn
import numpy as np
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


class bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, first = False):
        super(bottleneck, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.first = first
        self.stride = 1
        if self.first:
            self.stride = 2

        self.conv1 = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.mid_channels,
            kernel_size=1,
        )
        self.batch1 = nn.BatchNorm2d(num_features=mid_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels = self.mid_channels,
            out_channels = self.mid_channels,
            kernel_size = 3,
            stride = self.stride,
            padding = 1
        )
        self.batch2 = nn.BatchNorm2d(num_features=mid_channels)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=self.mid_channels,
            out_channels=self.out_channels,
            kernel_size=1
        )
        self.act3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(num_features=out_channels)
        self.proj = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride = self.stride), nn.BatchNorm2d(num_features=out_channels))
        self.net = nn.Sequential(self.conv1, self.batch1, self.act1, self.conv2, self.batch2, self.act2, self.conv3)

    def forward(self, x):
        x2 = self.net(x)
        x2 = self.batch3(x2)
        if self.first:
            x = self.proj(x)
        x = torch.add(x, x2)
        x = self.act3(x)
        return x



class ResNet101(nn.Module):
    def __init__(self, in_channels = 3, size = 224, classes = 10):
        super(ResNet101, self).__init__()
        self.classes = classes
        self.in_channels = in_channels
        self.size = size

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, stride=2, kernel_size=7, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Sequential(
            bottleneck(in_channels=64, mid_channels=64, out_channels=256, first=True),
            bottleneck(in_channels=256, mid_channels=64, out_channels=256),
            bottleneck(in_channels=256, mid_channels=64, out_channels=256),
        )

        self.conv3 = nn.Sequential(
            bottleneck(in_channels=256, mid_channels=128, out_channels=512, first=True),
            bottleneck(in_channels=512, mid_channels=128, out_channels=512),
            bottleneck(in_channels=512, mid_channels=128, out_channels=512),
            bottleneck(in_channels=512, mid_channels=128, out_channels=512),
        )
        conv4 = [bottleneck(in_channels=512, mid_channels=256, out_channels=1024, first=True)]
        conv4_2 = [bottleneck(in_channels=1024, mid_channels=256, out_channels=1024) for _ in range(22)]
        conv4.extend(conv4_2)
        self.conv4 = nn.Sequential(*conv4)

        self.conv5 = nn.Sequential(
            bottleneck(in_channels=1024, mid_channels=512, out_channels=2048, first=True),
            bottleneck(in_channels=2048, mid_channels=512, out_channels=2048),
            bottleneck(in_channels=2048, mid_channels=512, out_channels=2048),
        )

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(
            self.conv1,
            self.pool1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.pool2
        )
        self.flat = nn.Flatten()
        self.Linear = nn.Linear(in_features=2048, out_features=self.classes)
        self.soft = nn.Softmax(dim = 1)
        self.classifiers = nn.Sequential(
            self.Linear,
            self.soft
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flat(x)
        x = self.classifiers(x)

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

model = ResNet101(in_channels=3, size = 32, classes = 10)
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