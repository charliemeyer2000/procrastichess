#!/Users/charlie/miniconda3/bin/python3

import torch
from torch.utils.data import Dataset, DataLoader
import csv
import pandas as pd
import chess.pgn
import io
from board import Board

class CSVDataset(Dataset):
    def __init__(self):
        self.data = []

        df = pd.read_csv("data/club_games_data.csv")
        df = df.head(1)
        for _, row in df.iterrows():
            pgn = row['pgn']
            pgn_string = io.StringIO(pgn)
            game = chess.pgn.read_game(pgn_string)
            board = game.board()
            self.data.append(Board(board).serialize())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    csvDataset = CSVDataset()
    for board in csvDataset.data:
        print(board)
    
