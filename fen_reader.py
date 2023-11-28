#!/Users/charlie/miniconda3/bin/python3
import chess
from stockfish import Stockfish

class FenSupplier():
    def __init__(self, fen_file_path):
        self.fen_file_path = fen_file_path
        self.fen_list = self.initialize_fen_list()
    
    
    def initialize_fen_list(self):
        fen_list = []
        with open(self.fen_file_path, 'r') as f:
            for line in f:
                try:
                    fen = line.strip()
                    fen_list.append(fen)
                except:
                    continue
        return fen_list
    
    def get_first_fen(self):
        return self.fen_list[0]

    def get_fen_at_idx(self, idx):
        return self.fen_list[idx]


if __name__ == "__main__":
    stockfish = Stockfish("/usr/local/bin/stockfish")
    assert stockfish.is_fen_valid("r1bq1r1k/p1pnbpp1/1p2p3/6p1/3PB3/5N2/PPPQ1PPP/2KR3R")
