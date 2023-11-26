#!/Users/charlie/miniconda3/bin/python3
from torch.utils.data import Dataset, DataLoader
from CSVDataset import CSVDataset 
import numpy as np
import torch.nn.functional as F
import torch
import os


class ChessValueFuncNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = torch.nn.Linear(1024, 1)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 128 * 8 * 8)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # thanks geohot, -1 to 1 is better than 0 to 1
        return x

if __name__ == "__main__":
    # for apple silicon :)
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print("MPS is available")
    else:
        mps_device = torch.device("cpu")
        print("MPS is not available, using CPU")

    dataset = CSVDataset()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=32,
                              shuffle=True)
    model = ChessValueFuncNetwork().to(mps_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model.train()

    for epoch in range(100):
        all_loss = 0
        num_loss = 0
        for i, data in enumerate(train_loader):
            data = data.float().to(mps_device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, torch.zeros(len(output), device=mps_device))
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            num_loss += 1
            
        print("Epoch: {}, Loss: {}".format(epoch, all_loss / num_loss))
    if (os.path.isdir("output_nets") == False):
        os.mkdir("output_nets")

    torch.save(model.state_dict(), "output_nets/model.pth")

