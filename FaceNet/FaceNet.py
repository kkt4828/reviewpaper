from torch.nn.modules.distance import PairwiseDistance
from torchvision.models import resnet18
from torch.nn import functional as F

import torch.nn as nn
import torch
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



class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(p=2)

    def forward(self, anchor, pos, neg):
        pos_dist = self.pdist.forward(anchor, pos)
        neg_dist = self.pdist.forward(anchor, neg)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min = 0.0)
        loss = torch.mean(hinge_dist)

        return loss

class FaceNet_ResNet18(nn.Module):
    def __init__(self, embedding_dim = 128, pretrained=False):
        super(FaceNet_ResNet18, self).__init__()
        self.model = resnet18(pretrained=pretrained)

        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dim, bias=False)

    def forward(self, img):
        embedding = self.model(img)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

model = FaceNet_ResNet18()
criterion = nn.TripletMarginLoss(margin=0.2)

anc = torch.randn(32, 3, 224, 224)
pos = torch.randn(32, 3, 224, 224)
neg = torch.randn(32, 3, 224, 224)

a = model(anc)
p = model(pos)
n = model(neg)
loss = criterion(a, p, n)
print(loss)




