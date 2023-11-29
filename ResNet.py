from torchvision.models import resnet101
import torch.nn as nn
import torch


class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        self.resnet = resnet101()
        self.resnet.conv1 = nn.Conv2d(17, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        
    def forward(self, x):
        x = self.resnet(x)
        x = torch.tanh(x)
        return x