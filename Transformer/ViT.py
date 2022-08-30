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

class LinearPatchProjection(nn.Module):
    def __init__(self, device, batch_size, out_dim = 768, img_size=224, patch_size = 16, channel_size = 3,):
        super(LinearPatchProjection, self).__init__()
        self.device = device
        self.b = batch_size
        self.p = patch_size
        self.c = channel_size
        self.out_dim = out_dim
        self.n = img_size ** 2 // (patch_size ** 2)


        self.projection = nn.Linear(in_features=self.p**2 * self.c, out_features=self.out_dim)

    def forward(self, x):
        x = x.view(-1, self.n, (self.p ** 2) * self.c)
        x_p = self.projection(x)
        x_cls = nn.Parameter(torch.randn(x_p.size(0), 1, self.out_dim), requires_grad=True).to(device)
        x_pos = nn.Parameter(torch.randn(x_p.size(0), self.n + 1, self.out_dim), requires_grad=True).to(device)
        x_p = torch.concat((x_cls, x_p), dim = 1)
        x = torch.add(x_p, x_pos)

        return x

class SelfAttention(nn.Module):
    def __init__(self, out_dim = 768, d = 12):
        super(SelfAttention, self).__init__()
        self.out_dim = out_dim
        self.norm_scale = out_dim // d
        self.q = nn.Linear(in_features=out_dim, out_features=out_dim // d)
        self.k = nn.Linear(in_features=out_dim, out_features=out_dim // d)
        self.v = nn.Linear(in_features=out_dim, out_features=out_dim // d)
        self.soft = nn.Softmax(dim = -1)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        qk = torch.div(torch.matmul(q, torch.transpose(k, 1, 2)), self.norm_scale ** 0.5)
        qk = self.soft(qk)
        qkv = torch.matmul(qk, v)
        return qkv


class MultiHeadAttention(nn.Module):
    def __init__(self, out_dim = 768, h = 12):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.SA = nn.ModuleList([SelfAttention(out_dim, h) for _ in range(h)])
        self.linear = nn.Linear(in_features=out_dim, out_features=out_dim)

    def forward(self, x):
        for i in range(self.h):
            if i == 0:
                x_cat = self.SA[i](x)
            else:
                x_cat = torch.cat((x_cat, self.SA[i](x)), dim = -1)
        x = self.linear(x_cat)
        return x


class Encoder(nn.Module):
    def __init__(self, out_dim = 768, h = 12):
        super(Encoder, self).__init__()
        self.norm1 = nn.LayerNorm(out_dim)
        self.act1 = nn.GELU()
        self.mha = MultiHeadAttention(out_dim, h = h)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act2 = nn.GELU()
        self.linear = nn.Linear(in_features=out_dim, out_features=out_dim)

    def forward(self, x):
        x_norm = self.norm1(x)
        x_norm = self.act1(x_norm)
        x_norm = self.mha(x_norm)
        x = torch.add(x_norm, x)
        x_norm = self.norm2(x)
        x_norm = self.act2(x_norm)
        x_norm = self.linear(x_norm)
        x = torch.add(x_norm, x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, device, L = 12, out_dim = 768, h = 12, ML = 3096, num_classes = 10, img_size=224, patch_size = 16, channel_size = 3, batch_size = 16):
        super(VisionTransformer, self).__init__()
        self.batch_size = batch_size
        self.embedding = LinearPatchProjection(device, self.batch_size, out_dim, img_size, patch_size, channel_size)
        self.transencoder = nn.Sequential(*[Encoder(out_dim, h) for _ in range(L)])
        self.flatten = nn.Flatten()
        self.mlphead = nn.Sequential(nn.Linear(((img_size // patch_size) ** 2 + 1) * out_dim , num_classes))
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transencoder(x)
        x = self.flatten(x)
        x = self.mlphead(x)
        x = self.soft(x)

        return x

transform = transforms.Compose([
    # transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64

train_set = torchvision.datasets.CIFAR10(root = '../data', train = True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR10(root = '../data', train = False, transform=transform, download=True)

train_loader = DataLoader(train_set, shuffle=True, num_workers=1, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=False, num_workers=1, batch_size=batch_size)

EPOCHS = 20
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device('cuda')

model = VisionTransformer(device=device, batch_size=batch_size, out_dim = 144, img_size=32, patch_size=4)
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



