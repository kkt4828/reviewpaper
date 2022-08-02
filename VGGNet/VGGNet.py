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


def conv(stride = 1, padding = 1, kernel_size = 3, in_channels = 3, out_channels = 64):
    conv_l = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
    batch_1 = nn.BatchNorm2d(num_features = out_channels)
    act_1 = nn.ReLU()
    return nn.Sequential(conv_l, batch_1, act_1)


class VGGNet(nn.Module):
    def __init__(self, layers = 16, in_channels = 3, num_classes = 10, size = 224):
        """
        :param layers: 16 or 19 (number of layers)
        :param size: input size
        """
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = layers
        self.size = size

        self.conv1 = nn.Sequential(
            conv(stride = 1, padding = 1, kernel_size=3, in_channels = 3, out_channels=64),
            conv(stride = 1, padding = 1, kernel_size=3, in_channels = 64, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            conv(stride=1, padding=1, kernel_size=3, in_channels=64, out_channels=128),
            conv(stride=1, padding=1, kernel_size=3, in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        if self.layers == 16:
            self.conv3 = nn.Sequential(
                conv(stride=1, padding=1, kernel_size=3, in_channels=128, out_channels=256),
                conv(stride=1, padding=1, kernel_size=3, in_channels=256, out_channels=256),
                conv(stride=1, padding=1, kernel_size=3, in_channels=256, out_channels=256),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv4 = nn.Sequential(
                conv(stride=1, padding=1, kernel_size=3, in_channels=256, out_channels=512),
                conv(stride=1, padding=1, kernel_size=3, in_channels=512, out_channels=512),
                conv(stride=1, padding=1, kernel_size=3, in_channels=512, out_channels=512),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv5 = nn.Sequential(
                conv(stride=1, padding=1, kernel_size=3, in_channels=512, out_channels=512),
                conv(stride=1, padding=1, kernel_size=3, in_channels=512, out_channels=512),
                conv(stride=1, padding=1, kernel_size=3, in_channels=512, out_channels=512),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.conv3 = nn.Sequential(
                conv(stride=1, padding=1, kernel_size=3, in_channels=128, out_channels=256),
                conv(stride=1, padding=1, kernel_size=3, in_channels=256, out_channels=256),
                conv(stride=1, padding=1, kernel_size=3, in_channels=256, out_channels=256),
                conv(stride=1, padding=1, kernel_size=3, in_channels=256, out_channels=256),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv4 = nn.Sequential(
                conv(stride=1, padding=1, kernel_size=3, in_channels=256, out_channels=512),
                conv(stride=1, padding=1, kernel_size=3, in_channels=512, out_channels=512),
                conv(stride=1, padding=1, kernel_size=3, in_channels=512, out_channels=512),
                conv(stride=1, padding=1, kernel_size=3, in_channels=512, out_channels=512),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv5 = nn.Sequential(
                conv(stride=1, padding=1, kernel_size=3, in_channels=512, out_channels=512),
                conv(stride=1, padding=1, kernel_size=3, in_channels=512, out_channels=512),
                conv(stride=1, padding=1, kernel_size=3, in_channels=512, out_channels=512),
                conv(stride=1, padding=1, kernel_size=3, in_channels=512, out_channels=512),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.features = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)
        self.flat = nn.Flatten()
        self.classifiers = nn.Sequential(
            nn.Linear(in_features=512 * size * size // ((2 ** 5) * (2 ** 5)), out_features=4096, bias=True),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.Linear(in_features=4096, out_features=self.num_classes, bias=True),
        )
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.features(x)
        x = self.flat(x)
        x = self.classifiers(x)
        x = self.soft(x)

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

model = VGGNet(16, 3, 10, 32)
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