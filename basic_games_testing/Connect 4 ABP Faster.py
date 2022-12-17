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
        self.make_move_count = 0
        self.undo_move_count = 0
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
    def evaluate_position_direction(self, column, value, direction):
        # in essence tops marks the location of the next available square in a column, but we check this before a move has been made 
        # so though we are checking one above, we are pretending that a piece has been dropped there 

        position = (self.tops[column], column) # assume that the column has already been made
        
        if direction == 0 and position[0] <= NUM_ROWS - 3 - 1: #check up, no need to check down
            for i in range(1, 4):
                if position[0] + i >= NUM_ROWS:
                    return 0
                if self.board[position[0] + i, position[1]] != value:
                    return 0 #there is no splitting direction to check so just return 0    
            return 3
        if direction == 1: #check to the left
            count = 0
            for i in range(1, 4):
                if position[1] - i < 0 or  self.board[position[0], position[1] - i] != value:
                    break
                else:
                    count += 1
            for i in range(1, 4): #check to the right
                if count == 3 or position[1] + i >= NUM_COLS or  self.board[position[0], position[1] + i] != value:
                    break
                else:
                    count += 1
            return count
        if direction == 2:
            count = 0
            for i in range(1, 4):
                if position[0] - i < 0 or position[1] - i < 0 or  self.board[position[0] - i, position[1] - i] != value:
                    break
                else:
                    count+=1 
            for i in range(1, 4):
                if count == 3 or position[0] + i >= NUM_ROWS or position[1] + i >= NUM_COLS or  self.board[position[0] + i, position[1] + i] != value:
                    break
                else:
                    count+=1
            return count
        if direction == 3:
            count = 0
            for i in range(1, 4):
                if position[0] - i < 0 or position[1] + i >= NUM_COLS or self.board[position[0] - i, position[1] + i] != value:
                    break
                else:
                    count += 1
            for i in range(1, 4):
                if count == 3 or position[0] + i >= NUM_ROWS or position[1] - i < 0 or  self.board[position[0] + i, position[1] - i] != value:
                    break
                else:
                    count += 1
            return count
        return 0
    def evaluate_position(self, is_red, column):
        for i in range(4):
            if is_red:
                if self.evaluate_position_direction(column, RED, i) == 3:
                    return 1
            elif self.evaluate_position_direction(column, YELLOW, i) == 3:
                return -1
        return 0
    #based on our definition we can not check if a board is terminal only a column
    def is_terminal(self, column, isRed):
        if np.amax(self.tops) == 0 or self.evaluate_position(isRed, column) != 0:
            return True
        return False
    def make_move(self, column, isRed):
        self.make_move_count += 1
        if isRed:
            self.board[self.tops[column], column] = RED
            self.tops[column] -= 1
        else:
            self.board[self.tops[column], column] = YELLOW
            self.tops[column] -= 1
    def undo_move(self, column):
        self.undo_move_count += 1
        self.board[self.tops[column] + 1, column] = 0
        self.tops[column] += 1
    def evaluate_board(self, column, maximizingPlayer):
        if maximizingPlayer:
            max = -100
            for i in range(4):
                value = self.evaluate_position_direction(column, RED, i)
                if value > max:
                    max = value
            return max
        else:
            min = 10000
            for i in range(4):
                value = -1 * self.evaluate_position_direction(column, YELLOW, i)
                if value < min:
                    min = value
            return min

        return
    #maximizing player is the same as isRed(e.g. red is the maximizing player)
    def apb(self, depth, alpha, beta, maximizingPlayer):
        if depth == 1:
            if maximizingPlayer:
                max = -1000
                max_column = 0
                for column in [3,4,2,5,1,0,6]:
                    value = self.evaluate_board(column, maximizingPlayer)
                    if np.amax(self.tops) == 0 or value == 3:
                        return value, column
                    elif value > max:
                        max = value
                        max_column = column
                return max, max_column
            else:
                min = 1000
                min_column = 0
                for column in [3,4,2,5,1,0,6]:
                    value = self.evaluate_board(column, maximizingPlayer)
                    if np.amax(self.tops) == 0 or value == -3:
                        return value, column
                    elif value < min:
                        min = value
                        min_column = column
                return min, min_column        
        elif maximizingPlayer:
            max = -1000
            max_column = 0
            for column in [3,4,2,5,1,0,6]:
                if self.tops[column] < 0:
                    continue #ignore if we are in last column
                if self.is_terminal(column, maximizingPlayer):
                    return self.evaluate_board(column, maximizingPlayer), column
                self.make_move(column, True)
                value, _ = self.apb(depth - 1, alpha, beta, False)
                self.undo_move(column)
                if value > max:
                    max = value
                    max_column = column
                alpha = np.amax([alpha, value])
                if alpha >= beta:
                    break
            return max, max_column
        else:
            min = 1000
            min_column = 0
            for column in [3,4,2,5,1,0,6]:
                if self.tops[column] < 0:
                    continue #ignore if we are in last column
                if self.is_terminal(column, maximizingPlayer):
                    return self.evaluate_board(column, maximizingPlayer), column
                self.make_move(column, False)
                value, _ = self.apb(depth - 1, alpha, beta, True)
                self.undo_move(column)
                if value < min:
                    min = value
                    min_column = column
                beta = np.amin([beta, value])
                if beta <= alpha:
                    break
            return min, min_column
    def print_tops(self):
        for i in range(len(self.tops)):
            print(self.tops[i], ",", end="")
        print("")                

if __name__ == "__main__":
    play_first = int(input("0 to play first, 1 to play second, 2 for two player: "))
    if play_first == 0:
        board = ConnectFour(False) #initiate the board to 0   
        board.print_board() 
        while True:
            col = int(input("Red give a column: "))
            if board.is_terminal(col, True):
                board.make_move(col, True)
                break
            else:
                board.make_move(col, True)
            board.print_board()
            value, column = board.apb(9, -10000, 10000, False)
            if board.is_terminal(column, False):
                board.make_move(column, False)
                break
            else:
                board.make_move(column, False)
            print("board value: ", value)
            print("Column: ", column)
            print("Make move count: ", board.make_move_count)
            print("Undo move count: ", board.undo_move_count)
            board.print_tops()
            board.print_board()
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
    board.print_board()