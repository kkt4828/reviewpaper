import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

SEED = 2022

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    '''
    :param text: German text
    :return:  German text token list
    '''
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    :param text: English text
    :return: Englinh token list
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize = tokenize_de,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

TRG = Field(tokenize = tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
     batch_size = BATCH_SIZE,
     device = device)

# Encoder
# 기존 RNN과 다르게 input에 대한 output을 compress하지 않고 context vector sequence Z를 produce
# RNN은 input 이전 것 만 고려했다면 transformer에서는 모든 token가 관여됨
# embedding layer => Multi head attention => Feed Forward
# embedding layer에 recurrent가(순서를 고려하는 idea X) 없기때문에 대신 positional layer가 추가로 있음
# <sos>를 token에 시작에 추가로 넣었음 (position embedding : 단어 size 100으로 설정 => 조정 가능)
# 원래는 fixed static embedding 사용 => but 현대 BERT 등에서 사용하는 positional embedding 구현예정
# input embedding + positional embedding => scaling factor sqrt(d_model) => variance 감소를 위해 + 학습의 용이를 위해
# N 개의 encoder layer 통과 (Z를 얻기위해)
# src_mask : <pad> token에 대해서는 masking 해서 attention을 계산을 위해 필요

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim, # embedding length 조정 parameter
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim) # input embedding
        self.pos_embedding = nn.Embedding(max_length, hid_dim) # positional vector embedding

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)]) # encoder layer n번 통과
        self.dropout = nn.Dropout(dropout) # dropout

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device) # scale 조정


    def forward(self, src, src_mask):

        # src = [batch size, src len]
        # src_mask = [batch siz 1. 1. src len]
        batch_size = src.shape[0] # input의 batch size
        src_len = src.shape[1] # input token sequence 길이

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # position vector

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # input embedding 과 pos embedding 후 elementwise addition => scaling => dropout
        # encoder layer 통과 전 input 완성

        for layer in self.layers:
            src = layer(src, src_mask) # n번 encoder layer 통과

        return src # encoder 통과 후 output

# encoder layer는 encoder의 핵심 part
# src sentence 와 src mask 를 multi-head attention layer 통과 => src의 attention을 계산하는 layer
# 다른 sequence를 통해 attention을 계산하지 않고 자체 sentence에서 계산하므로 self attention이라 함
# dropout 수행
# residual connection + Layer Normalization 수행 => feature normalizing을 통해 layer가 많아도 쉽게 학습 가능하게 함
# position wise feedforward layer 통과
# dropout 수행
# resdiual connection + Layer Normalization 수행
# 다음 encoder layer로 보내줌


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src, src_mask):

        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # multihead attention 통과
        _src, _ = self.self_attention(src, src, src, src_mask)

        # residual connection + layer normailzation
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batchsize, src len, hid dim]

        # position-wise feedforward
        _src = self.positionwise_feedforward(src)

        # residual connection + layer normailzation
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batchsize, src len, hid dim]
        return src

# Multi Head Attention Layer
# Attention Q(queries), K(keys), V(values)로 계산됨
# Q(queries)는 K와 함께 attention vector를 얻을 때 활용됨(value에 가중합 시 사용)
# attention vector는 softmax를 통해 output을 만듦 (0~1 사잇값이며 모두 합하면 1)

# scaled dot-product attention 사용
# Q, K를 combine by dot product and scaling by d_k(head dim)
# Attention = softmax(QK^T/sqrt(d_k))V
# scaling을 통해 dot product에 의해 값이 커지는 것을 방지할 수 있음(이로 인하여 gradient가 너무 작아지는 것을 방지)

# 단순히 QKV를 scaled dotproduct attention 하지 않고
# Linear Transformation을 활용해 QKV 를 각각 h개(hid_dim)의 head로 split한 후 각각 계산 후 concat하는 방식을 채택

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim = -1)

        x = torch.matmul(self.dropout(attention), V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x, attention

# Position-wise Feedforward Layer
# encoder 또 다른 main block
# 논문에는 사용이유는 나와있지 않음
# BERT에서는 ReLU대신 GELU를 사용하는데 이것도 설명은 나와있지 않음

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.dropout(torch.relu(self.fc_1(x)))

        x = self.fc_2(x)

        return x

# Decoder
# encoder에서 encoded된 source를 predict token 으로 변환하는 부분
# encoder와 유사함
# Masked Multi-Head Attention layer / multi-head attention layer (decoder query + encoder key, value) 구조
# positional embeddings and combine(elementwise sum) them with scaled embedded target tokens
# positional encodings : 100 tokens long 까지 가능 (조정가능)
# encoded src => N decoder layers 통과 encoder / decoder layer 개수는 같을 필요는 없음
# N개의 decoder layer 통과 후 linear layer + softmax 통과

# <pad>를 모델에 넣지 않기위한 src mask 사용을 decoder에서도 똑같이 사용
# target mask도 사용함 => cheating 을 막기위해 사용함

class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)

        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        _trg = self.positionwise_feedforward(trg)

        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(self, trg):

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, trg):

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention


