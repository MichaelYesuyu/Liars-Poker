import random
import numpy as np

#For the sake of testing, computer will always play as X, and will go first
class Board:
    def __init__(self):
        self.board = np.zeros((3,3))
        self.move_list = []
        self.turn = 'X'

    def evaluate_position(self):
        cumsum = np.zeros(8)
        for i in range(3):
            cumsum[0] += board.board[i][i] #left diag
            cumsum[1] += board.board[2-i][i] #right diag
            cumsum[2] += board.board[0][i] #first row
            cumsum[3] += board.board[1][i] #second row
            cumsum[4] += board.board[2][i] #third row
            cumsum[5] += board.board[i][0] #first column
            cumsum[6] += board.board[i][1] #second column
            cumsum[7] += board.board[i][2] #third column
        max = np.amax(np.abs(cumsum))
        if not max == 3:
            return 0
        return np.sign(cumsum[np.argmax(np.abs(cumsum))])

    def is_terminal(self):
        if np.count_nonzero(board.board) == 9 or self.evaluate_position() != 0:
            return True
        else:
            return False

    def score(self):
        #This function returns the score based on who's turn it is, combines is_terminal and evaluate_position
        if self.evaluate_position() == 0:
            return 0
        elif self.evaluate_position() == 1:
            return -1 if self.turn == 'X' else 1
        else: #Else covers evaluate_position() == -1
            return -1 if self.turn == 'O' else 1

    #Pos is [y][x]
    def make_move(self, pos):
        if self.is_terminal():
            return
        if board.turn == 'X' and self.board[pos[0],pos[1]] == 0:
            self.board[pos[0],pos[1]] = 1
            board.turn = 'O'
        elif board.turn == 'O' and self.board[pos[0],pos[1]] == 0: 
            self.board[pos[0],pos[1]] = -1
            board.turn = 'X'
        self.move_list.append(pos)
    
    def undo_move(self):
        position_to_undo = self.move_list[-1]
        self.board[position_to_undo[0]][position_to_undo[1]] = 0
        self.move_list = self.move_list[:-1]
        if self.turn == 'X':
            self.turn = 'O'
        else:
            self.turn = 'X'

    def valid_moves(self):
        valid_moves_list = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    valid_moves_list.append((i,j))
        return valid_moves_list

    def print_board(self):
        for i in range(3):
            for j in range(3):
                print("|", end = "")
                if self.board[i][j] == 1:
                    print(" X |", end = "")
                elif self.board[i][j] == -1:
                    print(" O |", end = "")
                else:
                    print("   |", end="")
            print("")

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
def monte_carlo_value(board, N=1000):
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
    board = Board()
    while not board.is_terminal():
        board.print_board()
        print("-----------------")
        board.make_move(ai_best_move(board))
        board.print_board()
        if board.is_terminal():
            break
        print("-----------------")
        x = int(input("Give an x position: "))
        y = int(input("Give a y position: "))
        board.make_move((y,x))
        board.print_board()
        print("-----------------")