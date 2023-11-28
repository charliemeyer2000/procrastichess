#!/Users/charlie/miniconda3/bin/python3
from torch.utils.data import Dataset, DataLoader
from CSVDataset import CSVDataset 
from PGNDataset import PGNDataset
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import os


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


        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128, 1024)  
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        x = F.relu(self.conv7(x))
        x = self.bn7(x)
        x = F.relu(self.conv8(x))
        x = self.bn8(x)
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


    ###### INPUT DATA FOR NET GO HERE ######
    data_file_path = "data/club_games_data.csv"
    processed_file_path = "processed/processed_CSV_every_move_100klimit.pth"
    model_output_name = "model_50_epochs_everymove_100klimit_withbatchnorm_csv"
    ########################################


    print("Starting to load the data")
    dataset = CSVDataset(data_file_path=data_file_path, processed_file_path=processed_file_path)
    print("Finished loading the data")
    train_loader = DataLoader(dataset=dataset,
                              batch_size=32,
                              shuffle=True)
    model = ChessValueFuncNetwork().to(mps_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = torch.nn.MSELoss()

    model.train()

    num_epochs = 50
    print(f'Starting training with {num_epochs} epochs')
    if (os.path.isdir("output_nets") == False):
        os.mkdir("output_nets")
    if (os.path.isdir(f'output_nets/{model_output_name}') == False):
        os.mkdir(f'output_nets/{model_output_name}')

    for epoch in range(num_epochs):
        all_loss = 0
        num_loss = 0
        start_time = time.time()
        cur_loss = 0
        last_loss = 0
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
            scheduler.step()

            all_loss += loss.item()
            num_loss += 1
            end_time = time.time()
        print(f'Epoch: {epoch}, Loss: {all_loss / num_loss}, Time (in minutes): {(end_time - start_time) / 60}')
        cur_loss = all_loss / num_loss
        if (abs(cur_loss - last_loss) <= 0.0001):
            print("Loss has converged, stopping training")
            break
        last_loss = cur_loss
        epoch_dir = f'output_nets/{model_output_name}'
        # only save the model if the difference between the current loss and the last loss is greater than 0.1
        if (abs(cur_loss - last_loss) > 0.1):
            print(f'Saving model {model_output_name} at epoch {epoch}')
            torch.save(model.state_dict(), f'{epoch_dir}/EPOCH{epoch}_{model_output_name}.pth')

    torch.save(model.state_dict(), f'output_nets/{model_output_name}.pth')

