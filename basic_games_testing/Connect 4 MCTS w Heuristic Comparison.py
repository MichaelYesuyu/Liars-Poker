import numpy as np
import numpy.linalg as la
import time 


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
        self.simulations = {}
        self.simulations_after = {}
        self.values = {}
        self.win_count = {}
        self.was_terminal = {}
        self.C = 1.5
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
            max = -10000
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
    def update_value(self, column, value, wasTerminal):
        board_string = str(self.board.tobytes())
        hash_one = board_string + str(column)
        hash_two = board_string
        Ni = self.simulations.get(hash_two, 1) + 1
        n = self.simulations_after.get(hash_one, 0) + 1
        self.simulations[hash_two] = Ni
        self.simulations_after[hash_one] = n # this is wi in UCT formula 
        self.win_count[hash_one] = self.win_count.get(hash_one, 0) + value
        self.was_terminal[hash_one] = wasTerminal
        self.values[hash_one] = ((self.win_count.get(hash_one, 0) + value) / n) + self.C*np.sqrt(np.log(Ni) / n)
    def compute_value(self, column, isMaximizer):
        if self.tops[column] < 0:
            return -1 # don't make this move
        return self.values.get((str(self.board.tobytes()) + str(column)), self.C)
    def playout(self, isMaximizer):
        columns = np.array([3,4,2,5,1,0,6])
        hash_vals = [self.compute_value(i, isMaximizer) for i in columns]
        move = np.random.choice(columns[hash_vals == np.amax(hash_vals)])
        if self.is_terminal(move, isMaximizer):
            self.update_value(move, 1, True)
            return 1
        self.make_move(move, isMaximizer)
        value = -1*self.playout(not isMaximizer)
        self.undo_move(move)
        self.update_value(move, value, False)
        return value

    def MCTS(self, max_iters, maximizingPlayer):
        #start = time.perf_counter_ns()
        #duration = (time.perf_counter_ns() - start) // 1000000
        columns = np.array([3,4,2,5,1,0,6])
        #while duration < time:
        # Time seems to be the ideal implementation so I'll try and do that one later but for now I'll just do passes through the array
        for i in range(0,max_iters):
            hash_vals = [self.compute_value(col, maximizingPlayer) for col in columns]
            move = np.random.choice(columns[hash_vals == np.amax(hash_vals)])
            if self.is_terminal(move, maximizingPlayer):
                value = 1 # if we find a terminal move just play it no need to keep calculating
                self.update_value(move, value, True)
                break
            self.make_move(move, maximizingPlayer)
            value = -1 * self.playout(not maximizingPlayer)
            self.undo_move(move)
            self.update_value(move, value, False)
            #duration = (time.perf_counter_ns() - start) // 1000000
        max = -10000
        max_move = 0
        for move in range(0,7):
            value = self.values.get((str(self.board.tobytes()) + str(move)), -10)
            visits = self.simulations_after.get((str(self.board.tobytes()) + str(move)), -1)
            print("Column: ", move, " value: ", value,  " vists: ", visits)
            if value > max:
                max = value
                max_move = move
        return max, max_move
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
            value, column = board.MCTS(2000, False)
            if board.is_terminal(column, False):
                board.make_move(column, False)
                break
            else:
                board.make_move(column, False)
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