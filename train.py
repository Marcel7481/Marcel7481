import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_data=torchvision.datasets.CIFAR10(root="cifar10_data",train=True,transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data=torchvision.datasets.CIFAR10(root="cifar10_data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            # Conv2d(3, 32, 5, padding=2),
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        return self.model(x)

#初始化网络
network = NeuralNetwork()
#损失函数
loss_fn = nn.CrossEntropyLoss()
#优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(network.parameters(),lr=learning_rate)

total_train_step=0

total_test_step=0

epoch=10

writer = SummaryWriter("train_logs")
for i in range(epoch):
    # 训练
    print("Starting training epoch {}: ".format(i+1))
    step=0
    for data in train_dataloader:
        imgs, targets = data
        outputs=network(imgs)
        loss=loss_fn(outputs,targets)
       # 优化器优化模型
        optimizer.zero_grad()#将梯度清零
        loss.backward()
        optimizer.step()
        if total_train_step % 100 == 0:
            print("epoch:{}, step:{}. loss:{}".format(i+1,step,loss.item()))
            writer.add_scalar("train_loss",loss,total_train_step)
        total_train_step += 1
        step+=1
    # 测试
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = network(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
        print("totalLoss={}".format(total_test_loss))
        writer.add_scalar("totalLoss",total_test_loss,total_test_step)
        total_test_step+=1
writer.close()


