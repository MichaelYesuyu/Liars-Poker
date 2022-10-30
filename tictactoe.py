import numpy as np
import numpy.linalg as la

# min max search for tic tac toe, building up to McCFR
class Board:
    def __init__(self, isX):
        self.board = np.zeros((3, 3)) #board is numpy array, 1 represents x, -1 represents o 
        # player that is ai 
        # player that is human
        self.isX = True 
        if isX:
            self.player = 0 # 0 = x
            self.ai = 1 # 1 = 0
        else:
            self.player = 1
            self.ai = 0 
    def make_move(self, pos, isX):
        if self.is_terminal(self.board):
            return
        if isX and self.board[pos[0], pos[1]] == 0:
            self.board[pos[0], pos[1]] = 1
        elif self.board[pos[0], pos[1]] == 0: 
            self.board[pos[0], pos[1]] = -1
    # if three in a row for x, value is one, if three in a row for 0, value is negative one, else zero
    def evaluate_position(self, board):
        #check left diag
        cumsum = np.zeros(8)
        for i in range(3):
            cumsum[0] += board[i][i] #left diag
            cumsum[1] += board[2-i][i] #right diag
            cumsum[2] += board[0][i] #first row
            cumsum[3] += board[1][i] #second row
            cumsum[4] += board[2][i] #third row
            cumsum[5] += board[i][0] #first column
            cumsum[6] += board[i][1] #second column
            cumsum[7] += board[i][2] #third column
        max = np.amax(np.abs(cumsum))
        if not max == 3:
            return 0
        return np.sign(cumsum[np.argmax(np.abs(cumsum))])


            
    #Return true if board is in terminal state, else return false          
    def is_terminal(self, board):
        if np.count_nonzero(board) == 9 or self.evaluate_position(board) != 0:
            return True
        else:
            return False
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
    
    #Returns all possible child of node
    def get_children(self,node,isX):
        children = []
        if isX:
            for i in range(3):
                for j in range(3):
                    new_board = node.copy()
                    if node[i][j] == 0:
                        new_board[i][j] = 1
                        children.append(new_board)
        else:
            for i in range(3):
                for j in range(3):
                    new_board = node.copy()
                    if node[i][j] == 0:
                        new_board[i][j] = -1
                        children.append(new_board)
        return children
    def minmaxO(self, node, depth, maximizingPlayer):
        #if we are 0, then we want to minimize the resulting board
        #if we are X, then we want to maximize the resulting board
        if depth == 0 or self.is_terminal(node):
            return self.evaluate_position(node), node
        
        if maximizingPlayer:
            bestValue = 1000
            children = self.get_children(node,True)
            for child in children:
                v, _ = self.minimax(child, depth-1, False)
                if v > bestValue:
                    bestBoard = child
                    bestValue = v
            return bestValue, bestBoard
        
        else: #minimizing player
            bestValue = 1000
            children = self.get_children(node,False)
            for child in children:
                v, _ = self.minimax(child, depth-1, True)
                if v < bestValue:
                    bestBoard = child
                    bestValue = v 
            return bestValue, bestBoard
    #Node is a given board position
    def minimax(self, node, depth, maximizingPlayer):
        
        if depth == 0 or self.is_terminal(node):
            return self.evaluate_position(node), node
        
        if maximizingPlayer:
            bestValue = -1000
            children = self.get_children(node,True)
            for child in children:
                v, _ = self.minimax(child, depth-1, False)
                if v > bestValue:
                    bestBoard = child
                    bestValue = v
            return bestValue, bestBoard
        
        else: #minimizing player
            bestValue = 1000
            children = self.get_children(node,False)
            for child in children:
                v, _ = self.minimax(child, depth-1, True)
                if v < bestValue:
                    bestBoard = child
                    bestValue = v
            return bestValue, bestBoard

if __name__ == "__main__":
    play_first = int(input("0 to play first, 1 to play second: "))
    if play_first == 0:
        board = Board(False) #initiate the board to 0   
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
        board = Board(False) #initiate the board to 0   
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
    else:
        print("Type 1 or 0 only")

