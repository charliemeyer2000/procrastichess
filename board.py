#!/Users/charlie/miniconda3/bin/python3
import chess
import numpy as np

class Board():
    def __init__(self, board = None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def get_board(self):
        return self.board
    
    # convert to format for neural network, 8x8x14
    def serialize(self):
        assert self.board.is_valid
        # 8x8 squares, 12 piece types, 8x8 board
        bstate = np.zeros((12, 8, 8), np.uint8)

        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                piece_type = piece.piece_type - 1 
                color = piece.color
                index = piece_type + color * 6
                bstate[index, i // 8, i % 8] = 1

        # Encode castling rights
        castling_rights = np.zeros((4, 8, 8), np.uint8)
        if self.board.has_kingside_castling_rights(chess.WHITE):
            castling_rights[0] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            castling_rights[1] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            castling_rights[2] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            castling_rights[3] = 1

        # Encode en passant
        en_passant = np.zeros((1, 8, 8), np.uint8)
        if self.board.ep_square is not None:
            row_idx = self.board.ep_square // 8
            col_idx = self.board.ep_square % 8
            en_passant[0, row_idx, col_idx] = 1

        # Stack the channels together
        bstate = np.concatenate([bstate, castling_rights, en_passant])

        return bstate


if __name__ == "__main__": 
    s = Board()
    state = s.serialize()
    print(state.size)