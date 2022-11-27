from string import Formatter
import numpy as np
import numpy.linalg as la
import random
import pygame

class Move():
    def __init__(self, start_x, start_y, end_x, end_y, piece, is_black):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.piece = piece
        self.is_black = is_black
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
    #Consider whether it is better to have move as a tuple or as a class with a start end and piece variable
    def check_in_bounds(self, move):
        return (move.end_x < 8 and move.end_x >= 0 and move.end_y < 8 and move.end_y >= 0)
    def check_not_same(self, move):
        return (not move.end_x == move.start_x) or (not move.end_y == move.start_y) 
    def check_is_legal_move_king(self, move):
        return True
    def check_is_legal_move_queen(self, move):
        return True
    def check_is_legal_move_rook(self, move):
        return True
    def check_is_legal_move_pawn(self, move):
        return True
    def check_is_legal_move_bishop(self, move):
        return True
    def check_is_legal_move_knight(self, move):
        return True
    def is_legal_move(self, move):
        if not self.check_in_bounds(move):
            return False
        if not self.check_not_same(move):
            return False
        piece_temp = move.piece.lower()
        if piece_temp == 'k':
            return self.check_is_legal_move_king(move)
        if piece_temp == 'q':
            return self.check_is_legal_move_queen(move)
        if piece_temp == 'r':
            return self.check_is_legal_move_rook(move)
        if piece_temp == 'p':
            return self.check_is_legal_move_pawn(move)
        if piece_temp == 'b':
            return self.check_is_legal_move_bishop(move)
        if piece_temp == 'n':
            return self.check_is_legal_move_knight(move)
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
    def display_board(self, screen, surfaces_array):
        black_king, white_king, black_queen, white_queen, black_rook, white_rook, black_pawn, white_pawn, black_bishop, white_bishop, black_knight, white_knight = surfaces_array
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    color = (255, 255, 255)
                else:
                    color = (165,42,42)
                pygame.draw.rect(screen, color, pygame.Rect(i*60, j*60, 60, 60))
                if self.board[j][i] == 'O': #flip indices since j is row and i is column and board goes (row, column)
                    continue # do nothing
                elif self.board[j][i] == 'K': 
                    screen.blit(white_king, (i*60, j*60))
                elif self.board[j][i] == 'Q': 
                    screen.blit(white_queen, (i*60, j*60))
                elif self.board[j][i] == 'R': 
                    screen.blit(white_rook, (i*60, j*60))
                elif self.board[j][i] == 'P': 
                    screen.blit(white_pawn, (i*60, j*60))
                elif self.board[j][i] == 'B': 
                    screen.blit(white_bishop, (i*60, j*60))
                elif self.board[j][i] == 'N': 
                    screen.blit(white_knight, (i*60, j*60))
                elif self.board[j][i] == 'k': 
                    screen.blit(black_king, (i*60, j*60))
                elif self.board[j][i] == 'q': 
                    screen.blit(black_queen, (i*60, j*60))
                elif self.board[j][i] == 'r': 
                    screen.blit(black_rook, (i*60, j*60))
                elif self.board[j][i] == 'p': 
                    screen.blit(black_pawn, (i*60, j*60))
                elif self.board[j][i] == 'b': 
                    screen.blit(black_bishop, (i*60, j*60))
                elif self.board[j][i] == 'n': 
                    screen.blit(black_knight, (i*60, j*60))
    def get_board(self):
        return self.board


if __name__ == "__main__":
    pygame.init()
    chess = Chess()
    print(chess.print_board())
    background_color = (255,255,255)
    screen = pygame.display.set_mode((480, 480))


    #load images: 
    black_king = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\Chess_Engine\\images\\black_king.png")
    white_king = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\Chess_Engine\\images\\white_king.png")
    black_queen = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\Chess_Engine\\images\\black_queen.png")
    white_queen = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\Chess_Engine\\images\\white_queen.png")
    black_rook = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\Chess_Engine\\images\\black_rook.png")
    white_rook = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\Chess_Engine\\images\\white_rook.png")
    black_pawn = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\Chess_Engine\\images\\black_pawn.png")
    white_pawn = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\Chess_Engine\\images\\white_pawn.png")
    black_bishop = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\Chess_Engine\\images\\black_bishop.png")
    white_bishop = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\Chess_Engine\\images\\white_bishop.png")
    black_knight = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\Chess_Engine\\images\\black_knight.png")
    white_knight = pygame.image.load("C:\\Users\\vrsha\\Documents\\GitHub\\Liars-Poker\\basic_games_testing\\Chess_Engine\\images\\white_knight.png")

    surfaces_array = [black_king, white_king, black_queen, white_queen, black_rook, white_rook, black_pawn, white_pawn, black_bishop, white_bishop, black_knight, white_knight]
    # Set the caption of the screen
    pygame.display.set_caption('Chess')
    
    # Fill the background colour to the screen
    screen.fill(background_color)
    
    chess.display_board(screen, surfaces_array)
        
    # Update the display using flip
    pygame.display.flip()
    
    # Variable to keep our game loop running
    running = True
    mouse_is_pressed = False
    while running:
        for event in pygame.event.get():
            
            # Check for QUIT event
            if mouse_is_pressed:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                curr_row = int(int(mouse_y) / 60)
                curr_col = int(int(mouse_x) / 60) 
                move = Move(start_row, start_col, curr_row, curr_col, start_piece, start_piece.islower())
                chess.display_board(screen, surfaces_array) 
                if chess.is_legal_move(move):
                    pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(curr_col*60, curr_row*60, 60, 60),  2)
                else:
                    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(curr_col*60, curr_row*60, 60, 60),  2)
                pygame.display.flip()     
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_is_pressed = True
                mouse_x, mouse_y = pygame.mouse.get_pos()
                start_row = int(int(mouse_y) / 60)
                start_col = int(int(mouse_x) / 60) 
                start_piece = chess.get_board()[start_row][start_col]
                mouse_is_pressed = True
                
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                end_row = int(int(mouse_y) / 60) 
                end_col = int(int(mouse_x) / 60)
                chess.get_board()[end_row][end_col] = start_piece
                chess.get_board()[start_row][start_col] = "O"
                chess.display_board(screen, surfaces_array)
                pygame.display.flip()
                mouse_is_pressed = False
