
from string import Formatter
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
    #Let R always go first?
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
        print("---------------------------------")
    
    def evaluate_board(self, board):
      #Check rightwards facing diagonal
      for i in range(3):
          for j in range(4):
              sum = board[i][j] + board[i+1][j+1] + board[i+2][j+2] + board[i+3][j+3]
              if np.abs(sum) == 4:
                  return np.sign(sum)

      #Check leftwards facing diagonal
      for i in range(3):
          for j in range(3, 7, 1):
              sum = board[i][j] + board[i+1][j-1] + board[i+2][j-2] + board[i+3][j-3]
              if np.abs(sum) == 4:
                  return np.sign(sum)
      
      #Check horizontal
      for i in range(6):
          for j in range(4):
              sum = board[i][j] + board[i][j+1] + board[i][j+2] + board[i][j+3]
              if np.abs(sum) == 4:
                  return np.sign(sum)

      #Check vertical
      for i in range(3):
          for j in range(7):
              sum = board[i][j] + board[i+1][j] + board[i+2][j] + board[i+3][j]
              if np.abs(sum) == 4:
                  return np.sign(sum)
      #If nothing sums up to 4 or -4, return 0
      return 0

    def game_over(self, board):
        if self.evaluate_board(board) != 0 or np.count_nonzero(board) == NUM_COLS * NUM_ROWS:
            return True
        return False


    def get_children(self,node,isR):
        children = []
        if isR:
            for column in [3,4,2,5,1,6,0]:
                new_board = node.copy()
                for i in range(5,-1,-1):
                    if new_board[i][column] == 0:
                        new_board[i][column] = 1
                        children.append(new_board)
                        break
                    
        else:
            for column in [3,4,2,5,1,0,6]:
                new_board = node.copy()
                for i in range(5,-1,-1):
                    if new_board[i][column] == 0:
                        new_board[i][column] = -1
                        children.append(new_board)
                        break
                    

        return children

    def abpruning(self, node, depth, alpha, beta, maximizingPlayer):
        
        if depth == 0 or self.game_over(node):
            return self.evaluate_board(node), node
        
        if maximizingPlayer:
            bestValue = -1000
            children = self.get_children(node,True)
            for child in children:
                
                v, _ = self.abpruning(child, depth-1, alpha, beta, False)
                if v > alpha:
                    alpha = v
                if v > bestValue:
                    bestBoard = child
                    bestValue = v
                if beta <= alpha:
                    break
            return bestValue, bestBoard
        
        else: #minimizing player
            bestValue = 1000
            children = self.get_children(node,False) # get the Y children
            for child in children:
                v, _ = self.abpruning(child, depth-1, alpha, beta, True) # now have R pick its best move
                if v < beta:
                    beta = v
                if v < bestValue:
                    bestBoard = child
                    bestValue = v
                if beta <= alpha:
                    break
            return bestValue, bestBoard #recursively compute up the value tree

    def move(self, board, col, isRed):
        if isRed:
            for i in range(5,-1,-1):
                if board[i][col] == 0:
                    board[i][col] = 1
                    return board
        else:
            for i in range(5,-1,-1):
                if board[i][col] == 0:
                    board[i][col] = -1
                    return board


if __name__ == "__main__":
    play_first = int(input("0 to play first, 1 to play second, 2 for computer to play itself: "))
    #Would a structure like this work?
    if play_first == 0:
        board = ConnectFour(False) #initiate the board to 0   
        board.print_board() 
        print(board.game_over(board.board))
        while not board.game_over(board.board):
            board.print_board()
            col = int(input("Red give a column: "))
            board.board = board.move(board.board, col, True)
            value, board.board = board.abpruning(board.board, 8, -100, 100, False)
            board.print_board()
        board.print_board()
    elif play_first == 1:
        board = ConnectFour(False) #initiate the board to 0   
        board.print_board() 
        print(board.game_over(board.board))
        while not board.game_over(board.board):
            value, board.board = board.abpruning(board.board, 8, -100, 100, True)
            board.print_board()
            print("board value: ", value)
            if not board.game_over(board.board):
                col = int(input("Yellow give a column: "))
                board.board = board.move(board.board, col, False)
                #board.print_board()
           
        board.print_board()
    #This is not used
    elif play_first == 2:
        board = ConnectFour(False) #initiate the board to 0   
        board.print_board() 
        print(board.game_over(board.board))
        while not board.game_over(board.board):
            board.print_board()    
            value, board.board = board.abpruning(board.board, 8, -100, 100, True)
            value, board.board = board.abpruning(board.board, 8, -100, 100, False)
            board.print_board()
        board.print_board()
    else:
        print("Type 2 or 1 or 0 only")