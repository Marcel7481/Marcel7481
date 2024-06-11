import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, Sequential
from torch.utils.tensorboard import SummaryWriter


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.pool = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        #
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.flatten = Flatten()
        # self.fc1 = Linear(1024, 64)
        # self.fc2 = Linear(64, 10)

        # 使用Sequential
        self.model=Sequential(
            Conv2d(3, 32, 5, padding="same"),  #padding="same"自动计算对应的值，使得尺寸不变，这里的值为2
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        # x=self.conv1(x)
        # x=self.pool(x)
        # x=self.conv2(x)
        # x=self.pool(x)
        # x=self.conv3(x)
        # x=self.pool(x)
        # x=self.flatten(x)
        # x=self.fc1(x)
        # x=self.fc2(x)
        x=self.model(x)
        return x

writer = SummaryWriter("seq_logs")
network = NeuralNetwork()
print(network)
input = torch.ones((64, 3, 32, 32))
print(input.shape)
output = network(input)
print(output.shape)
writer.add_graph(network,input)
writer.close()
