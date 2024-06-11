import torch
X = torch.tensor([0,1,2,3,4])
dropout = 0.5
mask = (torch.randn(X.shape)> dropout).float()
X = mask * X / (1-dropout)
print(X)