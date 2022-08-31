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
    def __init__(self, out_dim = 768, img_size=224, patch_size = 4, channel_size = 3):
        super(LinearPatchProjection, self).__init__()
        self.p = patch_size
        self.c = channel_size
        self.out_dim = out_dim
        self.n = img_size ** 2 // (patch_size ** 2)
        self.projection = nn.Linear(in_features=self.p**2 * self.c, out_features=self.out_dim)

    def forward(self, x):
        x = x.view(-1, self.n, (self.p ** 2) * self.c)
        x_p = self.projection(x)

        return x_p

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
    def __init__(self, out_dim = 768, d = 12):
        super(MultiHeadAttention, self).__init__()
        self.d = d
        self.SA = nn.ModuleList([SelfAttention(out_dim, self.d) for _ in range(self.d)])
        self.linear = nn.Linear(in_features=out_dim, out_features=out_dim)

    def forward(self, x):
        for i in range(self.d):
            if i == 0:
                x_cat = self.SA[i](x)
            else:
                x_cat = torch.cat((x_cat, self.SA[i](x)), dim = -1)
        x = self.linear(x_cat)
        return x

class SBlock1(nn.Module):
    def __init__(self, in_dim = 768, out_dim = 768, d = 12, num_patch = 16, stage1 = False, patch_size = 4):
        super(SBlock1, self).__init__()
        self.stage1 = stage1
        self.out_dim = out_dim
        self.d = d
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.linear1 = nn.Linear(in_features=in_dim * 4, out_features=out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        if stage1:
            self.W_MSA = nn.ModuleList([MultiHeadAttention(out_dim, d) for _ in range(num_patch)])
        else:
            self.W_MSA = nn.ModuleList([MultiHeadAttention(out_dim, d) for _ in range(num_patch // 4)])
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x):
        if self.stage1:
            for i in range(self.num_patch):
                if i == 0:
                    x_p = x[:,:((self.patch_size) ** 2) * (i+1), :]
                    x_p = self.W_MSA[i](x_p)
                else:
                    x_p2 = x[:,((self.patch_size) ** 2) * (i):((self.patch_size) ** 2) * (i+1),:]
                    x_p = torch.concat((x_p, x_p2), dim=1)

        else:
            num = int(self.num_patch ** 0.5)
            x_selector = []
            for i in range(0, num, 2):
                for k in range(0, num, 2):
                    x_selector.append(torch.tensor([i * num + k, i * num + k + 1, (i + 1) * num + k, (i + 1) * num + 1 + k]))
            for j, idx in enumerate(x_selector):
                if j == 0:
                    print(idx)
                    x_p3 = torch.index_select(x, 1, idx)
                    b, _, f = x_p3.size()
                    x_p3 = x_p3.view(b, 1, 4 * f)
                else:
                    x_p2 = torch.index_select(x, 1, idx)
                    b, _, f = x_p2.size()
                    x_p2 = x_p2.view(b, 1, 4 * f)
                    x_p3 = torch.concat((x_p3, x_p2), dim = 1)
            x_p3 = self.linear1(x_p3)
            for i in range(self.num_patch // 4):
                if i == 0:
                    x_p = x_p3[:,:((self.patch_size) ** 2) * (i+1), :]
                    x_p = self.W_MSA[i](x_p)
                else:
                    x_p2 = x_p3[:,((self.patch_size) ** 2) * (i):((self.patch_size) ** 2) * (i+1),:]
                    x_p2 = self.W_MSA[i](x_p2)
                    x_p = torch.concat((x_p, x_p2), dim=1)

        return x_p


x = torch.randn(16, 3, 224, 224)
b = SBlock1()
em = LinearPatchProjection()

