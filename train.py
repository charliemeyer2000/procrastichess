#!/Users/charlie/miniconda3/bin/python3
from torch.utils.data import Dataset, DataLoader
from MyNets import ChessValueFuncNetwork
from CSVDataset import CSVDataset 
from PGNDataset import PGNDataset
import torch.nn.functional as F
from ResNet import MyResNet
import numpy as np
import torch.nn as nn
import torch
import time
import os

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
    processed_file_path = "processed/processed_CSV_every_move_500klimit.pth"
    model_output_name = "model_50_epochs_every_move_500klimit_csv_RESNET"
    ########################################


    print("Starting to load the data")
    dataset = CSVDataset(data_file_path=data_file_path, processed_file_path=processed_file_path)
    print("Finished loading the data")
    train_loader = DataLoader(dataset=dataset,
                              batch_size=32,
                              shuffle=True)
    model = MyResNet().to(mps_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
        least_loss_epoch = 999
        for i, (serialized_board, result) in enumerate(train_loader):
            serialized_board = serialized_board.float().to(mps_device)
            result = result.float().view(-1, 1).to(mps_device)
            optimizer.zero_grad()
            output = model(serialized_board)
            loss = loss_fn(torch.tanh(output), result)
            loss.backward()
            optimizer.step()
            scheduler.step()

            all_loss += loss.item()
            num_loss += 1
            end_time = time.time()

        print(f'Epoch: {epoch}, Loss: {all_loss / num_loss}, Time (minutes, hours): {(end_time - start_time) / 60}, {(end_time - start_time) / 3600}')
        cur_loss = all_loss / num_loss

        if (cur_loss < least_loss_epoch):
            least_loss_epoch = epoch

        if abs(cur_loss - last_loss) <= 0.0001:
            print("Loss has converged, stopping training")
            break

        print(f'Saving model {model_output_name} at epoch {epoch}')
        epoch_dir = f'output_nets/{model_output_name}'
        torch.save(model.state_dict(), f'{epoch_dir}/EPOCH{epoch}.pth')

        last_loss = cur_loss

    torch.save(model.state_dict(), f'output_nets/{model_output_name}.pth')
    print(f'Least loss was {least_loss_epoch} with loss {cur_loss}')


