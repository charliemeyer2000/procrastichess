#!/Users/charlie/miniconda3/bin/python3
from torch.utils.data import Dataset, DataLoader
from CSVDataset import CSVDataset 
from PGNDataset import PGNDataset
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import os


class ChessValueFuncNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(11, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, 1024)  
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.tanh(x)  
        return x

if __name__ == "__main__":
    # for apple silicon :)
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print("MPS is available")
    else:
        mps_device = torch.device("cpu")
        print("MPS is not available, using CPU")

    print("Starting to load the data")
    # dataset = PGNDataset("data/lichess_db.pgn")
    dataset = CSVDataset("processed/processed_CSV.pth")
    print("Finished loading the data")
    train_loader = DataLoader(dataset=dataset,
                              batch_size=32,
                              shuffle=True)
    model = ChessValueFuncNetwork().to(mps_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model.train()

    num_epochs = 100

    for epoch in range(num_epochs):
        all_loss = 0
        num_loss = 0
        for i, (serialized_board, result) in enumerate(train_loader):
            serialized_board = serialized_board.float().to(mps_device)
            # print(f'serialized_board.shape: {serialized_board.shape}')
            result = result.float().view(-1, 1).to(mps_device)  # Ensure result has the right shape
            # print(f'result.shape: {result.shape}')
            optimizer.zero_grad()
            output = model(serialized_board)
            # print(f'output.shape: {output.shape}')
            loss = loss_fn(torch.tanh(output), result)
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            num_loss += 1
        print("Epoch: {}, Loss: {}".format(epoch, all_loss / num_loss))
    if (os.path.isdir("output_nets") == False):
        os.mkdir("output_nets")

    torch.save(model.state_dict(), "output_nets/model_100_epochs_csvdata.pth")

