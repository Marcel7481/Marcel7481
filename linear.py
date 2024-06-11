import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

dataset = torchvision.datasets.CIFAR10("cifar10_data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=64,drop_last=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(196608,10)

    def forward(self,input):
        output=self.linear(input)
        return output

network = NeuralNetwork()

writer = SummaryWriter("linear_logs")
step=0
for data in dataloader:
    imgs, targets = data
    writer.add_image("raw", make_grid(imgs), global_step=step)
    # print(imgs.shape)
    input=torch.reshape(imgs,(1,1,1,-1))
    writer.add_image("input",make_grid(input),global_step=step)
    output=network(input)
    # print(output.shape)
    writer.add_image("output",make_grid(output),global_step=step)
    step+=1
writer.close()