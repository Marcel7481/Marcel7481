import torch
from torch import float32
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1,2,3],dtype=float32)
targets = torch.tensor([1,2,5],dtype=float32)
loss=L1Loss()
mse_loss=MSELoss()

print(loss(inputs,targets))
print(mse_loss(inputs,targets))


x=torch.tensor([0.2,0.7,0.1])
y=torch.tensor([1])
x=torch.reshape(x,(1,3))
cross_loss=CrossEntropyLoss()
print(cross_loss(x,y))