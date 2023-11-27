#!/Users/charlie/miniconda3/bin/python3
from torch.utils.data import Dataset
import chess.pgn
import subprocess
import io
import os
import torch
from board import Board

class PGNDataset(Dataset):
    def __init__(self, file_path, processed_file_path="processed/my_processed_data.pth"):
        if not os.path.exists("processed"):
            os.mkdir("processed")
        if os.path.exists(processed_file_path):
            self.data = torch.load(processed_file_path)
        else:
            self.data = []
            print("Loading data")
            self.load_data(file_path)
            torch.save(self.data, processed_file_path)

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            pgn_data = file.read()

        game = chess.pgn.read_game(io.StringIO(pgn_data))
        while game is not None:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            result = game.headers["Result"]
            if result == '1-0':
                result = 1
            elif result == '0-1':
                result = -1
            else:
                result = 0
            game = chess.pgn.read_game(io.StringIO(pgn_data))
            self.data.append((Board(board).serialize(), result))
            print("Loaded game", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
if __name__ == "__main__":
    file_path = "data/lichess_db.pgn"
    pgnDataset = PGNDataset(file_path=file_path)
    print(pgnDataset[0].shape)

