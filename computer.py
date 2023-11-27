#!/Users/charlie/miniconda3/bin/python3
from train import ChessValueFuncNetwork
from stockfish import Stockfish
from board import Board
import pandas as pd
import chess
import torch
import io

class ComputerValueFunction():
    def __init__(self):
        model = torch.load("output_nets/model.pth")
        self.model = ChessValueFuncNetwork()
        self.model.load_state_dict(model)

    def evaluate(self, board):
        board = board.serialize()
        output = self.model(torch.tensor(board).float())
        return output[0][0].item()
    
    def fen_to_board(fen):
        return chess.Board(fen)
    

# compare to stockfish
if __name__ == "__main__":
    fen = "r3k2r/p1pp1pb1/bn2Qnp1/2qPN3/1p2P3/2N5/PPPBBPPP/R3K2R b KQkq - 3 2"


    stockfish = Stockfish("/usr/local/bin/stockfish")
    stockfish.set_fen_position(fen)
    stockfish_eval = stockfish.get_evaluation()

    # compare to computer
    computer = ComputerValueFunction()
    board = Board(chess.Board(fen))
    computer_eval = computer.evaluate(board)

    print("Stockfish eval: ", stockfish_eval['value'] / 100)
    print("Computer eval: ", computer_eval)




    




