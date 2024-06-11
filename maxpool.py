import torch
from PIL import Image
from torch.nn import MaxPool2d
from torchvision import transforms
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
input = torch.reshape(input,(-1,1,5,5))
print(input)
print(input.shape)
maxpool1=MaxPool2d(kernel_size=3,ceil_mode=False)
output=maxpool1(input)
print(output)
