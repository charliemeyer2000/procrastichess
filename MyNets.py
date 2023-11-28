from torch.nn import functional as F
import torch.nn as nn
import torch

class ChessValueFuncNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(17, 64, kernel_size=3, padding=1) # 12 + 4 + 1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)


        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(128)
        # self.bn5 = nn.BatchNorm2d(128)
        # self.bn6 = nn.BatchNorm2d(128)
        # self.bn7 = nn.BatchNorm2d(128)
        # self.bn8 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128, 1024)  
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.bn1(x)
        x = F.relu(self.conv2(x))
        # x = self.bn2(x)
        x = F.relu(self.conv3(x))
        # x = self.bn3(x)
        x = F.relu(self.conv4(x))
        # x = self.bn4(x)
        x = F.relu(self.conv5(x))
        # x = self.bn5(x)
        x = F.relu(self.conv6(x))
        # x = self.bn6(x)
        x = F.relu(self.conv7(x))
        # x = self.bn7(x)
        x = F.relu(self.conv8(x))
        # x = self.bn8(x)
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.tanh(x)  
        return x