#!/Users/charlie/miniconda3/bin/python3
from torch.utils.data import Dataset
import chess.pgn
import subprocess
import io
import os
import torch
from board import Board

class PGNDataset(Dataset):
    def __init__(self, data_file_path, processed_file_path="processed/processed_pgn.pth"):
        if not (processed_file_path.endswith(".pth")):
            raise Exception("processed_file_path must end with .pth")
        if not (data_file_path.endswith(".zst")):
            raise Exception("data_file_path must end with .zst")
        if not os.path.exists("processed"):
            os.mkdir("processed")
        if os.path.exists(processed_file_path):
            print("Already processed data exists at", processed_file_path, " , loading that instead")
            self.data = torch.load(processed_file_path)
        else:
            self.data = []
            print("Loading data")
            self.load_data(data_file_path)
            torch.save(self.data, processed_file_path)

    def load_data(self, file_path):
        with open(file_path, encoding="utf-8-sig", errors="ignore") as pgn:
            while True:
                try:
                    game = chess.pgn.read_game(pgn)
                except Exception as e:
                    print(e)
                    continue
                board = game.board()

                for move in game.mainline_moves():
                    try:
                        board.push(move)
                    except Exception as e:
                        print(e)
                        continue

                    result = game.headers["Result"]
                    if result == '1-0':
                        result = 1
                    elif result == '0-1':
                        result = -1
                    else:
                        result = 0
                    self.data.append((Board(board).serialize(), result))
                print("Loaded game", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
if __name__ == "__main__":
    data_path = "data/lichess_db_standard_rated_2016-11.pgn.zst"
    processed_file_path = "processed/lichess_db_standard_rated_2016-11.pth"
    pgnDataset = PGNDataset(data_file_path=data_path, processed_file_path=processed_file_path)
    print(pgnDataset[0][0].shape)

