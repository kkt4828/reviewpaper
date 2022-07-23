import torch
import torch.nn as nn
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, sampler
import torch.optim as optim

# import numpy as np
# import os
# import copy
# import cv2
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

class image_embedding(nn.Module):
    def __init__(self, in_channels = 3, img_size = 224, patch_size = 16, emb_dim = 16 * 16 * 3):
        super().__init__()

        self.rearrange = Rearrange('b c (num_w p1) (num_h p2) -> b (num_w num_h) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.linear = nn.Linear(in_channels * patch_size * patch_size, emb_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        n_patches = img_size * img_size // patch_size ** 2
        self.positions = nn.Parameter(torch.randn(n_patches + 1, emb_dim))


    def forward(self, x):
        batch, channel, width, height = x.shape
        # print('before rearrange x shape :', x.shape)
        x = self.rearrange(x)
        # print('after rearrange x shape :', x.shape)
        x = self.linear(x)
        # print('cls_token shape :', self.cls_token.shape)
        c = repeat(self.cls_token, '() n d -> b n d', b = batch)
        x = torch.cat((c, x), 1)
        # print('positions shape :', self.positions.shape)
        x = torch.add(x, self.positions)

        return x

# x = torch.randn(1, 3, 224, 224)
# emb = image_embedding(3, 224, 16, 16 * 16 * 3)(x)
# print(emb.shape)

class multi_head_attention(nn.Module):
    def __init__(self, emb_dim : int = 16 * 16 * 3, num_heads : int = 8, dropout_ratio : float = 0.2, verbose = False, **kwargs):
        super(multi_head_attention, self).__init__()
        self.v = verbose

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scaling = (self.emb_dim // num_heads) ** (-0.5)

        self.value = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)
        self.att_drop = nn.Dropout(dropout_ratio)

        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, x : Tensor) -> Tensor:
        # query, key, value

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        if self.v: print(Q.size(), K.size(), V.size())

        # q = k = v = patch_size * 2 + 1 & h * d = emb_dim
        Q = rearrange(Q, 'b q (h d) -> b h q d', h = self.num_heads)
        K = rearrange(K, 'b k (h d) -> b h d k', h = self.num_heads)
        V = rearrange(V, 'b v (h d) -> b h v d', h = self.num_heads)
        if self.v: print(Q.size(), K.size(), V.size())

        # scaled dot-product
        weight = torch.matmul(Q, K)
        weight = weight * self.scaling
        if self.v: print(weight.size())

        attention = torch.softmax(weight, dim = -1)
        attention = self.att_drop(attention)
        if self.v: print(attention.size())

        context = torch.matmul(attention, V)
        context = rearrange(context, 'b h q d -> b q (h d)')
        if self.v: print(context.size())

        x = self.linear(context)
        return x, attention

# feat, att = multi_head_attention(16 * 16 * 3, 8, verbose = True)(emb)
# print('feat shape: ', feat.shape)
# print('att shape: ', att.shape)

class mlp_block(nn.Module):
    def __init__(self, emb_dim : int = 16 * 16 * 3, forward_dim : int = 4, dropout_ratio : float = 0.2, **kwargs):
        super(mlp_block, self).__init__()
        self.linear_1 = nn.Linear(emb_dim, forward_dim * emb_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear_2 = nn.Linear(emb_dim * forward_dim, emb_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x


class encoder_block(nn.Module):
    def __init__(self, emb_dim : int = 16 * 16 * 3, num_heads : int = 8, forward_dim : int = 4, dropout_ratio : float = 0.2):
        super(encoder_block, self).__init__()
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.mha = multi_head_attention(emb_dim, num_heads, dropout_ratio)

        self.norm_2 = nn.LayerNorm(emb_dim)
        self.mlp = mlp_block(emb_dim, forward_dim, dropout_ratio)

        self.residual_dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x2 = self.norm_1(x)
        x2, attention = self.mha(x2)

        x = torch.add(x2, x)

        x2 = self.norm_2(x)
        x2 = self.mlp(x2)
        x = torch.add(x2, x)

        return x, attention

# feat, att = encoder_block()(emb)
# print('feat shape:', feat.shape)
# print('att shape: ', att.shape)

class vision_transformer(nn.Module):
    def __init__(self, in_channels : int = 3, img_size : int = 224, patch_size : int = 16, emb_dim : int = 16 * 16 * 3,
                 n_enc_layers : int = 15, num_heads : int = 3, forward_dim : int = 4, dropout_ratio : float = 0.2, n_classes : int = 1000 ):
        super(vision_transformer, self).__init__()

        self.image_embedding = image_embedding(in_channels, img_size, patch_size, emb_dim)
        encoder_module = [encoder_block(emb_dim, num_heads, forward_dim, dropout_ratio) for _ in range(n_enc_layers)]
        self.encoder_module = nn.ModuleList(encoder_module)

        self.reduce_layer = Reduce('b n e -> b e', reduction = 'mean')
        self.normalization = nn.LayerNorm(emb_dim)
        self.classification_head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        x = self.image_embedding(x)
        attentions = [block(x)[1] for block in self.encoder_module]

        x = self.reduce_layer(x)
        x = self.normalization(x)
        x = self.classification_head(x)

        return x, attentions

# y, att = vision_transformer(3, 224, 32, 3 * 32 * 32, 15, 3, 4, 0.2, 10)(x)
# print('y shape: ', y.shape, 'attentions[0] shape: ', att[0].shape)
# print(y)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 8

train_set = torchvision.datasets.CIFAR10(root = '../data', train = True, download = True, transform = transform)
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 1)

test_set = torchvision.datasets.CIFAR10(root = '../data', train = False, download = True, transform = transform)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device('cuda')

model = vision_transformer(3, 32, 4, 3 * 4 * 4, 15, 3, 4, 0.2, 10)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.0005)
epochs = 2
if __name__ == '__main__':
    for epoch in range(epochs):
        running_loss = 0.0
        t_count = 0
        a_count = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs, attentions = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            t_count += batch_size
            outputs2 = torch.argmax(outputs, 1)
            a_count += int(sum(outputs2 == labels))

            if i % 2000 == 1999:
                print(f'{epoch + 1} {i + 1} loss : {running_loss / (i + 1)} acc : {a_count / t_count}')
        print(f'loss : {running_loss / t_count} acc : {a_count / t_count}')

    print('Finish')
