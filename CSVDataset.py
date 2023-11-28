#!/Users/charlie/miniconda3/bin/python3

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import chess.pgn
import io
import os
from board import Board

class CSVDataset(Dataset):
    def __init__(self, data_file_path, processed_file_path="processed/processed_CSV.pth"):
        if not os.path.exists("processed"):
            os.mkdir("processed")
        if os.path.exists(processed_file_path):
            print("Already processed data exists at", processed_file_path, " , loading that instead")
            self.data = torch.load(processed_file_path)
        else:
            self.data = []
            self.load_data(data_file_path)
            torch.save(self.data, processed_file_path)

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            if (len(self.data) > 100_000):
                print("Loaded 100k games, stopping.")
                break
            pgn = row['pgn']
            pgn_string = io.StringIO(pgn)
            game = chess.pgn.read_game(pgn_string)
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                if row['black_result'] == 'win':
                    result = -1
                elif row['white_result'] == 'win':
                    result = 1
                else:
                    result = 0
                self.data.append((Board(board).serialize(), result))
            
            print("Loaded game", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    csvDataset = CSVDataset(processed_file_path="processed/processed_CSV_every_move_100klimit.pth", data_file_path="data/club_games_data.csv")
    print(f'In total there are {len(csvDataset)} games')
    print(csvDataset[0][0].shape)
    