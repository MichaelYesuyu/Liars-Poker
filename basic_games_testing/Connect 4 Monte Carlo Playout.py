from string import Formatter
import numpy as np
import numpy.linalg as la
import random

NUM_ROWS = 6
NUM_COLS = 7
RED = 1 
YELLOW = -1 

class ConnectFour():
    def __init__(self):
        self.board = np.zeros((NUM_ROWS,NUM_COLS))
        self.turn = 'R' #Red always starts first
        self.move_list = []

    def print_board(self):
        for i in range(NUM_ROWS):
            print ("|", end = "")
            for j in range(NUM_COLS):
                if self.board[i][j] == RED:
                    print(" R |", end = "")
                elif self.board[i][j] == YELLOW:
                    print(" Y |", end = "")
                else:
                    print("   |", end = "")
            print("")
        print("-------------------------------")

    def evaluate_board(self):
        #Check rightwards facing diagonal
        for i in range(3):
            for j in range(4):
                sum = self.board[i][j] + self.board[i+1][j+1] + self.board[i+2][j+2] + self.board[i+3][j+3]
                if np.abs(sum) == 4:
                    return np.sign(sum)

        #Check leftwards facing diagonal
        for i in range(3):
            for j in range(3, 7, 1):
                sum = self.board[i][j] + self.board[i+1][j-1] + self.board[i+2][j-2] + self.board[i+3][j-3]
                if np.abs(sum) == 4:
                    return np.sign(sum)
        
        #Check horizontal
        for i in range(6):
            for j in range(4):
                sum = self.board[i][j] + self.board[i][j+1] + self.board[i][j+2] + self.board[i][j+3]
                if np.abs(sum) == 4:
                    return np.sign(sum)

        #Check vertical
        for i in range(3):
            for j in range(7):
                sum = self.board[i][j] + self.board[i+1][j] + self.board[i+2][j] + self.board[i+3][j]
                if np.abs(sum) == 4:
                    return np.sign(sum)
        #If nothing sums up to 4 or -4, return 0
        return 0

    def is_terminal(self):
        if self.evaluate_board() != 0 or np.count_nonzero(self.board) == NUM_COLS * NUM_ROWS:
            return True
        return False

    def valid_moves(self):
        valid_moves_list = []
        for column in range(7):
            for row in range(6):
                if self.board[row][column] == 0:
                    valid_moves_list.append(column)
                    break
        return valid_moves_list

    def score(self): #Need to return -1 if it's not the agent's turn and they just lost
        if self.evaluate_board() == 0:
            return 0
        elif self.evaluate_board() == 1:
            return -1 if self.turn == 'R' else 1
        else:
            return -1 if self.turn == 'Y' else 1

    def make_move(self, column):
        if self.is_terminal():
            return
        if self.turn == 'R':
            for row in range(5,-1,-1):
                if self.board[row][column] == 0:
                    self.board[row][column] = 1
                    self.turn = 'Y'
                    break
        elif self.turn == 'Y':
            for row in range(5,-1,-1):
                if self.board[row][column] == 0:
                    self.board[row][column] = -1
                    self.turn = 'R'
                    break
        self.move_list.append(column)   

    def undo_move(self):
        column_to_undo = self.move_list[-1]
        for row in range(6):
            if self.board[row][column_to_undo] != 0:
                self.board[row][column_to_undo] = 0
                self.move_list = self.move_list[:-1]
                break
        if self.turn == 'R':
            self.turn = 'Y'
        else:
            self.turn = 'R'

def playout_value(board):
    if board.is_terminal():
        return -1 * board.score()
    move = random.choice(board.valid_moves())
    board.make_move(move)
    value = -1 * playout_value(board)
    board.undo_move()
    return value

"""
Finds the expected value of a game by running the specified number
of random simulations.
"""
def monte_carlo_value(board, N=200):
    scores = [playout_value(board) for i in range(0, N)]
    return np.mean(scores)

"""
Chooses best valued move to play using Monte Carlo Tree search.
"""
def ai_best_move(board):
    action_dict = {}
    for move in board.valid_moves():
        board.make_move(move)
        action_dict[move] = -1 * monte_carlo_value(board)
        board.undo_move()
        print(board.valid_moves())
        print(action_dict)
    return max(action_dict, key=action_dict.get)   

if __name__ == "__main__":
    board = ConnectFour()
    mode = int(input("1 to play first, 2 to play second: "))
    if mode == 1:
        while not board.is_terminal():
            board.print_board()
            col = int(input("Give a column: "))
            board.make_move(col)
            board.print_board()
            if board.is_terminal():
                break
            board.make_move(ai_best_move(board))
            board.print_board()
    elif mode == 2:
        while not board.is_terminal():
            board.print_board()
            board.make_move(ai_best_move(board))
            board.print_board()
            if board.is_terminal():
                break
            col = int(input("Give a column: "))
            board.make_move(col)
            board.print_board()
    else:
        print("Please enter 1 or 2")


      