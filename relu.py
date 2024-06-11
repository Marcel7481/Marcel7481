import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# input = torch.tensor([[1, -0.5],
#                       [-1, 3]])
# input = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10("cifar10_data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=64)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        # output = self.relu(input)
        output=self.sigmoid(input)
        return output

writer = SummaryWriter("relu_logs")
step=0
network = NeuralNetwork()
for data in dataloader:
    imgs, targets = data
    writer.add_image("input",make_grid(imgs),global_step=step)
    output = network(imgs)
    writer.add_image("output",make_grid(output),global_step=step)
    step = step + 1
writer.close()


