import torch

x=torch.tensor([[1,2],
                [0,4]])

print(x.argmax(dim=1))#横
print(x.argmax(dim=0))#纵