import numpy as np
import matplotlib as plt

class ConnectFourBoard:
    def __init__(self):
        self.board = np.zeros((6,7)) # 6 rows 7 columns
        self.