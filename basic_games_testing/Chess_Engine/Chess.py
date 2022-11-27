from string import Formatter
import numpy as np
import numpy.linalg as la
import random
import pygame
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
                    continue
        return 0
        #Gets both legal and nonlegal moves, and use get_legal_moves to filter out, for each legal move, 

    def get_legal_moves(self):
        return 0
        #Under get_legal_moves there should be individual functions checking based on what the piece is

    def is_in_check(self):
        return 0
        #Checks to see if board position is in check, should it take in board in addition to self?
    
    def make_move(self):
        return 0

    def undo_move(self):
        return 0

    def is_checkmate(self):
        return 0

    def is_draw(self):
        return 0

    def is_terminal(self):
        return 0
    def display_board(self, screen):
        #do nothing for now
        return 0
    def get_board(self):
        return self.board


if __name__ == "__main__":
    pygame.init()
    chess = Chess()
    print(chess.print_board())
    background_color = (255,255,255)
    screen = pygame.display.set_mode((480, 480))
    black_king = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\ChessEngine\\images\\black_king.png")
    white_king = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\ChessEngine\\images\\white_king.png")
    black_queen = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\ChessEngine\\images\\black_queen.png")
    white_queen = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\ChessEngine\\images\\white_queen.png")
    black_rook = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\ChessEngine\\images\\black_rook.png")
    white_rook = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\ChessEngine\\images\\white_rook.png")
    black_pawn = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\ChessEngine\\images\\black_pawn.png")
    
    white_pawn = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\ChessEngine\\images\\white_pawn.png")
    black_bishop = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\ChessEngine\\images\\black_bishop.png")
    white_bishop = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\ChessEngine\\images\\white_bishop.png")
    black_knight = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\ChessEngine\\images\\black_knight.png")
    white_knight = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\ChessEngine\\images\\white_knight.png")
    # Set the caption of the screen
    pygame.display.set_caption('Chess')
    
    # Fill the background colour to the screen
    screen.fill(background_color)

    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                color = (255, 255, 255)
            else:
                color = (0,0,0)
            pygame.draw.rect(screen, color, pygame.Rect(i*60, j*60, 60, 60))
            if chess.get_board()[i][j] == 'O':
                continue # do nothing
            elif chess.get_board()[i][j] == 'K': 
                screen.blit(white_king, (i*60, j*60))
            elif chess.get_board()[i][j] == 'Q': 
                screen.blit(white_queen, (i*60, j*60))
            elif chess.get_board()[i][j] == 'R': 
                screen.blit(white_rook, (i*60, j*60))
            elif chess.get_board()[i][j] == 'P': 
                screen.blit(white_pawn, (i*60, j*60))
            elif chess.get_board()[i][j] == 'B': 
                screen.blit(white_bishop, (i*60, j*60))
            elif chess.get_board()[i][j] == 'N': 
                screen.blit(white_knight, (i*60, j*60))
            elif chess.get_board()[i][j] == 'k': 
                screen.blit(black_king, (i*60, j*60))
            elif chess.get_board()[i][j] == 'q': 
                screen.blit(black_queen, (i*60, j*60))
            elif chess.get_board()[i][j] == 'r': 
                screen.blit(black_rook, (i*60, j*60))
            elif chess.get_board()[i][j] == 'p': 
                screen.blit(black_pawn, (i*60, j*60))
            elif chess.get_board()[i][j] == 'b': 
                screen.blit(black_bishop, (i*60, j*60))
            elif chess.get_board()[i][j] == 'n': 
                screen.blit(black_knight, (i*60, j*60))    
    # Update the display using flip
    pygame.display.flip()
    
    # Variable to keep our game loop running
    running = True
