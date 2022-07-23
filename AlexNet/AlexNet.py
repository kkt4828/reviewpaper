import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, sampler
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

class Alexnet(nn.Module):
    def __init__(self, in_channels = 3, classes = 1000):
        super().__init__()
        self.classes = classes
        self.in_channels = in_channels
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels = self.in_channels,
                                               out_channels = 96,
                                               kernel_size = 11,
                                               stride = 4,
                                               padding = 0),
                                     nn.BatchNorm2d(num_features=96),
                                     nn.ReLU(),
                                     nn.Dropout2d(),
                                     nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels = 96,
                                               out_channels = 256,
                                               kernel_size = 5,
                                               stride = 1,
                                               padding = 2),
                                     nn.BatchNorm2d(num_features=256),
                                     nn.ReLU(),
                                     nn.Dropout2d(),
                                     nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels = 256,
                                               out_channels = 384,
                                               kernel_size = 3,
                                               stride = 1,
                                               padding = 1),
                                     nn.BatchNorm2d(num_features=384),
                                     nn.ReLU(),
                                     nn.Dropout2d())
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels = 384,
                                               out_channels = 384,
                                               kernel_size = 3,
                                               stride = 1,
                                               padding = 1),
                                     nn.BatchNorm2d(num_features=384),
                                     nn.ReLU(),
                                     nn.Dropout2d())
        self.conv_5 = nn.Sequential(nn.Conv2d(in_channels = 384,
                                               out_channels = 256,
                                               kernel_size = 3,
                                               stride = 1,
                                               padding = 1),
                                     nn.BatchNorm2d(num_features=256),
                                     nn.ReLU(),
                                     nn.Dropout2d(),
                                     nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(in_features = 6 * 6 * 256, out_features = 4096)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features = 4096, out_features = 4096)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features = 4096, out_features = self.classes)
        self.soft = nn.Softmax(dim = -1)


    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.soft(x)
        return x



transforms = transforms.Compose(
    [transforms.Resize(227),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 8

train_data = torchvision.datasets.CIFAR10(root = '../data', train = True, transform = transforms, download = True)
train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size, num_workers = 1)

test_data = torchvision.datasets.CIFAR10(root = '../data', train = False, transform = transforms, download = True)
test_loader = DataLoader(test_data, shuffle = False, batch_size = batch_size, num_workers = 1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device('cuda')
model = Alexnet(classes = 10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params = model.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.0005)

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
                print(f"val loss : {val_loss:.5f} val acc : {val_acc:.3f}")






