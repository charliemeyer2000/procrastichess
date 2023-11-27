#!/Users/charlie/miniconda3/bin/python3
# from computer import ChessValueFuncNetwork
# import numpy as np
# import chess.engine
# import chess
# import torch

# class UCIPlayer():


import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")

board = chess.Board()

while not board.is_game_over():

    result = engine.play(board, chess.engine.Limit(time=0.1))
    board.push(result.move)
    print(board)

engine.quit()