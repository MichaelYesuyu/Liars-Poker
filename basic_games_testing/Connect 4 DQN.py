import numpy as np
import numpy.linalg as la
import time 
import tensorflow as tf
import random
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation

from keras.optimizers import Adam

NUM_ROWS = 6
NUM_COLS = 7
NUM_CHANNELS = 1
MIN_MEM_SIZE = 1000
RED = 1 
YELLOW = -1 
GAMMA = 0.9
UPDATES = 10 # update our network once every 10 steps 
MEMORY_SIZE = 5000
MINI_BATCH_SIZE = 200
REPLACE_COUNT = 7
EPS_DECAY = 0.99995
EPS_MIN = 0.1
EPS = 1

class ConnectFour:
    def __init__(self, start_red = True):
        print(EPS)
        self.board = np.zeros((1, NUM_ROWS,NUM_COLS, NUM_CHANNELS))
        self.tops = (np.ones(NUM_COLS, dtype=np.int64) * NUM_ROWS) - 1 # this tells us what the current open index is for each column
        self.start_red = start_red
        self.training_model = self.build_model()
        self.active_model = self.build_model()
        self.memory = self.build_memory()
        self.updates = 0 # for 
        self.iterations = 0
        self.epsilon = EPS
    def clear(self):
        self.board = np.zeros((1,NUM_ROWS,NUM_COLS, NUM_CHANNELS))
        self.tops = (np.ones(NUM_COLS, dtype=np.int64) * NUM_ROWS) - 1
    def save_model(self):
        self.active_model.save('connect_four_model_dql')
    def load_model(self):
        self.active_model = tf.keras.models.load_model('connect_four_model_dql')
        self.training_model.set_weights(self.active_model.get_weights())  
    def build_memory(self): 
        memory = []
        for i in range(MEMORY_SIZE):
            memory.append((np.zeros(self.board.shape), 0, 0, np.zeros(self.board.shape), False))
        return memory
    def build_model(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(64, (4,4), input_shape = (NUM_ROWS, NUM_COLS, NUM_CHANNELS), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (4,4), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Dense(64))
        model.add(Dense(7, activation='linear'))
        model.compile(loss = "mse", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        return model
    #basically our goal here is to compute a minibatch of q values against model predictions
    def train(self):
        if self.iterations < MIN_MEM_SIZE:
            return
        if not (self.iterations % 100 == 0):
            return # basically just break out here 
        print(self.iterations)
        if self.iterations < MEMORY_SIZE:
            new_memory = random.sample(self.memory[:self.iterations], MINI_BATCH_SIZE)
        else:
            new_memory = random.sample(self.memory, MINI_BATCH_SIZE)
        current_states = np.array([memory[0] for memory in new_memory]).reshape((MINI_BATCH_SIZE, NUM_ROWS, NUM_COLS, NUM_CHANNELS))
        current_qs = self.training_model.predict(current_states, verbose = 0)
        new_states = np.array([memory[3] for memory in new_memory]).reshape((MINI_BATCH_SIZE, NUM_ROWS, NUM_COLS, NUM_CHANNELS))
        new_qs = self.active_model.predict(new_states, verbose = 0)
        X = []
        Y = []
        for index, (current_state, action, reward, new_state, terminal) in enumerate(new_memory):
            q_new = reward
            if not terminal:
                q_new += GAMMA * np.amax(new_qs[index])
            q_new = np.min([np.abs(q_new), 1]) * np.sign(q_new)
            q_current = current_qs[index]
            q_current[action] = q_new
            X.append(current_state)
            Y.append(q_current)
        X = np.array(X).reshape((MINI_BATCH_SIZE, NUM_ROWS, NUM_COLS, NUM_CHANNELS))
        Y = np.array(Y)
        self.training_model.fit(X, Y, verbose = 0, epochs=10)
        self.updates += 1
        if self.updates % REPLACE_COUNT == 0:
            self.active_model.set_weights(self.training_model.get_weights())
            print("updating model")
            self.save_model()

    #assume current board hasn't been won
    #check if adding a new piece at given location will result in a win
    def invert_perspective(self):
        np.place(self.board, self.board == YELLOW, [127])
        np.place(self.board, self.board == RED, [YELLOW])
        np.place(self.board, self.board == 127, [RED])
    def print_board(self):
        for i in range(NUM_ROWS):
            print ("|", end = "")
            for j in range(NUM_COLS):
                if self.board[0][i][j] == RED:
                    print(" R |", end = "")
                elif self.board[0][i][j] == YELLOW:
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
                if self.board[0,position[0] + i, position[1]] != value:
                    return 0 #there is no splitting direction to check so just return 0    
            return 3
        if direction == 1: #check to the left
            count = 0
            for i in range(1, 4):
                if position[1] - i < 0 or  self.board[0,position[0], position[1] - i] != value:
                    break
                else:
                    count += 1
            for i in range(1, 4): #check to the right
                if count == 3 or position[1] + i >= NUM_COLS or  self.board[0,position[0], position[1] + i] != value:
                    break
                else:
                    count += 1
            return count
        if direction == 2:
            count = 0
            for i in range(1, 4):
                if position[0] - i < 0 or position[1] - i < 0 or  self.board[0,position[0] - i, position[1] - i] != value:
                    break
                else:
                    count+=1 
            for i in range(1, 4):
                if count == 3 or position[0] + i >= NUM_ROWS or position[1] + i >= NUM_COLS or  self.board[0,position[0] + i, position[1] + i] != value:
                    break
                else:
                    count+=1
            return count
        if direction == 3:
            count = 0
            for i in range(1, 4):
                if position[0] - i < 0 or position[1] + i >= NUM_COLS or self.board[0,position[0] - i, position[1] + i] != value:
                    break
                else:
                    count += 1
            for i in range(1, 4):
                if count == 3 or position[0] + i >= NUM_ROWS or position[1] - i < 0 or  self.board[0,position[0] + i, position[1] - i] != value:
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
    def is_terminal(self, column, is_red):
        if np.amax(self.tops) == 0 or self.evaluate_position(is_red, column) != 0:
            return True
        return False
    def make_move(self, column, is_red):
        if is_red:
            self.board[0,self.tops[column], column] = RED
            self.tops[column] -= 1
        else:
            self.board[0,self.tops[column], column] = YELLOW
            self.tops[column] -= 1
    def undo_move(self, column):
        self.board[0,self.tops[column] + 1, column] = 0
        self.tops[column] += 1
    def evaluate_board(self, column, is_red):
        if is_red:
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
    def compute_value(self):
        out = self.active_model.predict(self.board, verbose = 0)[0]
        return out
    def update_memory(self, input_tuple):
        if self.iterations < MEMORY_SIZE:
            self.memory[self.iterations % MEMORY_SIZE] = input_tuple
        else: 
            index = np.random.randint(MEMORY_SIZE)
            self.memory[index] = input_tuple # randomly replace an entry
        self.iterations += 1
        if self.iterations % MEMORY_SIZE == 0:
            print("iterations mod mem equals zero")
        self.train()
    """
    Plays an episode of Connect Four for training 
    Uses Epsilon Greedy to get a move from the active network 
    """
    def NN_episode_train(self):
        self.invert_perspective()
        columns = np.array([3,4,2,5,1,0,6])
        tops_mask = [self.tops[column] >= 0 for column in columns]
        model_q_values = self.compute_value()
        columns = columns[tops_mask]
        if(len(columns) == 0):
            print("board has been filled!")
            return 0
        move = columns[np.argmax([model_q_values[col] for col in columns])]
        self.epsilon *= EPS_DECAY
        self.epsilon = max(self.epsilon, EPS_MIN)
        if np.random.rand() < self.epsilon:
            move = np.random.choice(columns)
        if self.is_terminal(move, True):
            reward = 1
            start_board = self.board
            self.make_move(move, True)
            end_board = self.board
            self.undo_move(move)
            input_tuple = (start_board, move, reward, end_board, True)
            self.update_memory(input_tuple)
            return reward
        else:
            start_board = self.board
            self.make_move(move, True)
            self.invert_perspective()
            reward = 0
            move_enem = columns[np.argmax([model_q_values[col] for col in columns])]
            is_terminal = False
            if self.is_terminal(move_enem, True):
                reward = -1
                is_terminal = True
            self.make_move(move_enem, True)
            self.invert_perspective()
            end_board = self.board
            self.undo_move(move_enem)
            self.NN_episode_train()
            self.undo_move(move)
            input_tuple = (start_board, move, reward, end_board, is_terminal)
            self.update_memory(input_tuple)
            return 0
        
    def NN_move_test(self, is_red):

        columns = np.array([3,4,2,5,1,0,6])
        tops_mask = [self.tops[column] >= 0 for column in columns]
        model_q_values = self.compute_value()
        print("model q values: ", model_q_values)
        columns = columns[tops_mask]
        print("columns: ", columns)
        if is_red:
            move = columns[np.argmax([model_q_values[col] for col in columns])]
        else:
            self.invert_perspective() 
            move = columns[np.argmax([model_q_values[col] for col in columns])]
            self.invert_perspective()
        return move

    def print_tops(self):
        for i in range(len(self.tops)):
            print(self.tops[i], ",", end="")
        print("")                

if __name__ == "__main__":
    train_or_test = int(input("0 to train, 1 to test: "))
    if train_or_test == 0:
        board = ConnectFour(False)
        num_iters = int(input("enter the number of iterations that you want to train for: "))
        for i in range(num_iters):
            if i % 100 == 0:
                print("board epsilon: ", board.epsilon)
                print("iteration: ", i)
            board.clear()
            board.invert_perspective()
            board.NN_episode_train() # we always start as red
        board.save_model()
    if train_or_test == 1:
        play_first = int(input("0 to play first, 1 to play second, 2 for two player: "))
        if play_first == 0:
            board = ConnectFour(False) #initiate the board to 0   
            board.load_model()
            board.clear()
            board.print_board() 
            while True:
                col = int(input("Red give a column: "))
                if board.is_terminal(col, True):
                    board.make_move(col, True)
                    break
                else:
                    board.make_move(col, True)
                board.print_board()
                column = board.NN_move_test(False)
                if board.is_terminal(column, False):
                    board.make_move(column, False)
                    break
                else:
                    board.make_move(column, False)
                board.print_board()
            board.print_board()
        elif play_first == 1:
            board = ConnectFour(False) #initiate the board to 0   
            board.load_model()
            board.clear()
            board.print_board() 
            while True:
                column = board.NN_move_test(True)
                if board.is_terminal(column, True):
                    board.make_move(column, True)
                    break
                else:
                    board.make_move(column, True)
                board.print_board()
                col = int(input("Yellow give a column: "))
                if board.is_terminal(col, False):
                    board.make_move(col, False)
                    break
                else:
                    board.make_move(col, False)
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