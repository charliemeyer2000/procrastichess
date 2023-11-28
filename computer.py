#!/Users/charlie/miniconda3/bin/python3
from stockfish import StockfishException
from train import ChessValueFuncNetwork
import matplotlib.pyplot as plt
from fen_reader import FenSupplier
from stockfish import Stockfish
from ResNet import MyResNet
from board import Board
import pandas as pd
import numpy as np
import chess
import torch
import io

class ComputerValueFunction():
    def __init__(self, model_path = "output_nets/model.pth"):
        self.model = MyResNet()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()  # Set the model to evaluation mode

    def evaluate(self, board):
        board = Board(board)
        board = board.serialize()
        output = self.model(torch.tensor(board).unsqueeze(0).float())
        return output[0][0].item()
    
    def fen_to_board(fen):
        return chess.Board(fen)
    

def normalize_stockfish_eval(eval_sf):
    return max(-1, min(1, eval_sf / 1000))


    

# compare to stockfish
if __name__ == "__main__":
    print("loading fen supplier")
    fenSupplier = FenSupplier("data/432k-chess-puzzles-sorted.fen")
    print('loaded fen supplier')
    fen_list = fenSupplier.fen_list

    stockfish = Stockfish("/usr/local/bin/stockfish")
    path_to_model = "output_nets/model_50_epochs_every_move_csv_RESNET/EPOCH4.pth"
    computer = ComputerValueFunction(model_path=path_to_model)

    stockfish_evals = []
    computer_evals = []
    avg_diff = 0
    differences = []

    print("starting evaluations")

    num_games_evaluated = 0
    for index, fen in enumerate(fen_list):
        if "(" in fen or ")" in fen:
            continue
        try:
            # if not stockfish.is_fen_valid(fen):
            #     print(f"Invalid FEN {fen}")
            #     continue
            stockfish.set_fen_position(fen)
            stockfish_eval = stockfish.get_evaluation()
            if stockfish_eval is None:
                print(f"Game over for FEN {fen}")
                continue
            stockfish_eval = normalize_stockfish_eval(stockfish_eval["value"])
        except StockfishException:
            print(f"Stockfish exception for FEN {fen}")
            continue

        # compare to computer
        board = chess.Board(fen)
        if not board.is_valid():
            print(f"Invalid board for FEN {fen}")
            continue
        computer_eval = computer.evaluate(board)

        # Printing
        print("--------------------------------")
        print(f"stockfish_eval: {stockfish_eval}")
        print(f"computer_eval: {computer_eval}")
        print(f'Difference: {abs(stockfish_eval - computer_eval)}')

        # Incrementors
        stockfish_evals.append(stockfish_eval)
        computer_evals.append(computer_eval)
        diff = abs(stockfish_eval - computer_eval)
        differences.append(diff)
        avg_diff += diff
        num_games_evaluated += 1

    

    print("--------------------------------")
    print(f"Average difference: {avg_diff / len(fen_list)}")
    print(f'Number of games evaluated: {num_games_evaluated}')
    print(f'Standard deviation: {np.std(differences)}')

    ### SO far epoch 26 is the best



    




