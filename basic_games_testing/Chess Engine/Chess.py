from string import Formatter
import numpy as np
import numpy.linalg as la
import random

class Chess():
    def __init__(self):
        self.board = self.create_new_board()
        self.turn = 'W'
        self.white_pieces = "RNBQKP"
        self.black_pieces = "rnbqkp"
        self.move_list = [] #Maybe store the previous board? If so how to calculate/undo? Perhaps need to store 2 things, beginning and end square, as well as entire board position
        self.board_history = []

    def create_new_board(self):
        new_board = np.full((8,8),'O')
        new_board[0][0], new_board[0][7] = 'r', 'r'
        new_board[0][1], new_board[0][6] = 'n', 'n'
        new_board[0][2], new_board[0][5] = 'b', 'b'
        new_board[0][3] = 'q'
        new_board[0][4] = 'k'
        new_board[1] = 'p'
        new_board[6] = 'P'
        new_board[7][0], new_board[7][7] = 'R', 'R'
        new_board[7][1], new_board[7][6] = 'N', 'N'
        new_board[7][2], new_board[7][5] = 'B', 'B'
        new_board[7][3] = 'Q'
        new_board[7][4] = 'K'
        return new_board

    def print_board(self):
        for row in range(8):
            print(str(np.abs(row - 8)) + " ", end = "")
            for col in range(8):
                print("|", end = "")
                piece = self.board[row][col]
                if piece != 'O':
                    print(" " + piece + " |", end = "")
                else:
                    print("   |", end = "")
            print("")
            print("   ----------------------------------------")
        print("    A    B    C    D    E    F    G    H")
        print("-------------------------------------------")

    """
    Special rules that need to be handled:
    1. Promotion
    2. En Passant
    3. Castling

    Other items:
    1. Checking for checkmate
    2. Checking for 3 fold repetition
    3. Checking for 50 move rule draw
    """

    #List of functions needed:
    def get_all_moves(self):
        all_moves_list = []
        #don't think this works cuz you need to reorient the board based on the player
        if self.turn == 'W':
            check_pieces = self.white_pieces
        else:
            check_pieces = self.black_pieces
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece == 'O':
                    continue
                #"RNBQKP"
                elif piece == check_pieces[0]: #Rook



                

        return
        #Gets both legal and nonlegal moves, and use get_legal_moves to filter out, for each legal move, 

    def get_legal_moves(self):
        return
        #Under get_legal_moves there should be individual functions checking based on what the piece is

    def is_in_check(self):
        return
        #Checks to see if board position is in check, should it take in board in addition to self?
    
    def make_move(self):
        return

    def undo_move(self):
        return

    def is_checkmate(self):
        return

    def is_draw(self):
        return

    def is_terminal(self):
        return

chess = Chess()
print(chess.print_board())


