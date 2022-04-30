import torch

x = torch.tensor([[1,2,3,4], [5, 6, 7, 8]])
print(x.unsqueeze(1).shape)
print(x.unsqueeze(1).unsqueeze(2).shape)