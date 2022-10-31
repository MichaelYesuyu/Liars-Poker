import numpy as np
import numpy.linalg as la

NUM_ROWS = 6
NUM_COLS = 7
RED = 1 
YELLOW = -1 
class ConnectFour:
    def __init__(self, start_red = True):
        self.board = np.zeros((NUM_ROWS,NUM_COLS))
        self.tops = (np.ones(NUM_COLS, dtype=np.int64) * NUM_ROWS) - 1 # this tells us what the current open index is for each column
        self.start_red = start_red
    #assume current board hasn't been won
    #check if adding a new piece at given location will result in a win
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
    def evaluate_position_direction(self, board, position, value, direction):
        if direction == 0 and position[0] <= NUM_ROWS - 3 - 1: #check up, no need to check down
            for i in range(1, 4):
                if position[0] + i >= NUM_ROWS:
                    return 0
                if board[position[0] + i, position[1]] != value:
                    return 0 #there is no splitting direction to check so just return 0    
            return 3
        if direction == 1: #check to the left
            count = 0
            for i in range(1, 4):
                if position[1] - i < 0 or  board[position[0], position[1] - i] != value:
                    break
                else:
                    count += 1
            for i in range(1, 4): #check to the right
                if count == 3 or position[1] + i >= NUM_COLS or  board[position[0], position[1] + i] != value:
                    break
                else:
                    count += 1
            print("direction one count: ", count)
            return count
        if direction == 2:
            count = 0
            for i in range(1, 4):
                if position[0] - i < 0 or position[1] - i < 0 or  board[position[0] - i, position[1] - i] != value:
                    break
                else:
                    count+=1 
            for i in range(1, 4):
                if count == 3 or position[0] + i >= NUM_ROWS or position[1] + i >= NUM_COLS or  board[position[0] + i, position[1] + i] != value:
                    break
                else:
                    count+=1
            print("direction two count: ", count)
            return count
        if direction == 3:
            count = 0
            for i in range(1, 4):
                if position[0] - i < 0 or position[1] + i >= NUM_COLS or board[position[0] - i, position[1] + i] != value:
                    break
                else:
                    count += 1
            for i in range(1, 4):
                if count == 3 or position[0] + i >= NUM_ROWS or position[1] - i < 0 or  board[position[0] + i, position[1] - i] != value:
                    break
                else:
                    count += 1
            print("direction three count: ", count)
            return count
        return 0
    def evaluate_position(self, board, is_red, position):
        for i in range(4):
            if is_red:
                if self.evaluate_position_direction(board, position, RED, i) == 3:
                    return 1
            elif self.evaluate_position_direction(board, position, YELLOW, i) == 3:
                return -1
        return 0
    #based on our definition we can not check if a board is terminal only a move
    def is_terminal(self, board, move, isRed):
        if np.amax(self.tops) == 0 or self.evaluate_position(board, isRed, move) != 0:
            return True
        return False
    def make_move(self, move, isRed):
        if isRed:
            self.board[self.tops[move], move] = RED
            self.tops[move] -= 1
        else:
            self.board[self.tops[move], move] = YELLOW
            self.tops[move] -= 1


if __name__ == "__main__":
    play_first = int(input("0 to play first, 1 to play second, 2 for two player: "))
    if play_first == 0:
        board = ConnectFour(False) #initiate the board to 0   
        board.print_board() 
        print(board.is_terminal(board.board))
        while not board.is_terminal(board.board):
            board.print_board()
            x = int(input("Give an x position: "))
            y = int(input("Give a y position: "))
            board.make_move((y,x), True)
            value, board.board = board.minimax(board.board, 9, False)
            print("board value: ", value)
        board.print_board()
    elif play_first == 1:
        board = ConnectFour(False) #initiate the board to 0   
        board.print_board() 
        print(board.is_terminal(board.board))
        while not board.is_terminal(board.board):
            value, board.board = board.minimax(board.board, 9, True)
            board.print_board()
            print("board value: ", value)
            if not board.is_terminal(board.board):
                x = int(input("Give an x position: "))
                y = int(input("Give a y position: "))
                board.make_move((y,x), False)
           
        board.print_board()
    elif play_first == 2:
        board = ConnectFour(True)
        while True:
            col = int(input("Red give a column: "))
            if board.is_terminal(board.board, (board.tops[col], col), True):
                board.make_move(col, True)
                break
            else:
                board.make_move(col, True)
            board.print_board()
            col = int(input("Yellow give a column: "))
            if board.is_terminal(board.board, (board.tops[col], col), False):
                board.make_move(col, False)
                break
            else:
                board.make_move(col, False)
            board.print_board()
        board.print_board()
    else:
        print("Type 1 or 0 only")