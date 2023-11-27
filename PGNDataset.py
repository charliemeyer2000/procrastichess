#!/Users/charlie/miniconda3/bin/python3
from torch.utils.data import Dataset
import chess.pgn
import subprocess
import io
from board import Board

class PGNDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        print("Loading data")
        self.load_data(file_path)

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            pgn_data = file.read()

        game = chess.pgn.read_game(io.StringIO(pgn_data))
        while game is not None:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                self.data.append(Board(board).serialize())
            game = chess.pgn.read_game(io.StringIO(pgn_data))
            print("Loaded game", len(self.data))
            
        print("Loaded data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
if __name__ == "__main__":
    file_path = "data/lichess_db.pgn"
    pgnDataset = PGNDataset(file_path=file_path)
    for board in pgnDataset.data:
        print(board.shape)
