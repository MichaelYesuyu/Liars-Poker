import numpy as np
import random
import matplotlib.pyplot as plt

NUM_EPISODES = 100000
NUM_ACTIONS = 3
ROCK = 0
PAPER = 1
SCISSORS = 2

class RockPaperScissorsTrainer():
    def __init__(self):
        self.strategy_profile_sum = np.zeros(NUM_ACTIONS)
        self.regret_sum = np.zeros(NUM_ACTIONS)
        self.num_iters = 0
        self.rock_regrets = []
        self.paper_regrets = []
        self.scissors_regrets = []
    def get_regret_strategy(self):
        regret_sum = np.sum(self.regret_sum[self.regret_sum > 0])
        if regret_sum <= 0:
            strategy = np.ones(NUM_ACTIONS) / NUM_ACTIONS
        else:
            strategy = np.zeros(NUM_ACTIONS)
            for i in range(NUM_ACTIONS):
                if(self.regret_sum[i] > 0):
                    strategy[i] = self.regret_sum[i] / regret_sum
        self.strategy_profile_sum += strategy
        self.num_iters += 1
        self.rock_regrets.append(self.regret_sum[ROCK])
        self.paper_regrets.append(self.regret_sum[PAPER])
        self.scissors_regrets.append(self.regret_sum[SCISSORS])
        return strategy
    def get_action(self, strategy):
        return np.random.choice([ROCK, PAPER, SCISSORS], p=strategy)
    def is_winner(self, p1_move, p2_move):
        if p1_move == ROCK and p2_move == SCISSORS:
            return True
        if p1_move == SCISSORS and p2_move == PAPER:
            return True
        if p1_move == PAPER and p2_move == ROCK:
            return True
        return False
    def compute_regrets_one_sided(self, player_move, oponent_move):
        if player_move == ROCK and oponent_move == ROCK: 
            self.regret_sum[PAPER] += 1 # both oponents regret tying
            self.regret_sum[SCISSORS] -= 1
        elif player_move == PAPER and oponent_move == PAPER:
            self.regret_sum[SCISSORS] += 1
            self.regret_sum[ROCK] -= 1
        elif player_move == SCISSORS and oponent_move == SCISSORS:
            self.regret_sum[ROCK] += 1
            self.regret_sum[PAPER] -= 1
        elif self.is_winner(oponent_move, player_move):
            self.compute_regrets_ordered(player_move, oponent_move)
        else:
            self.compute_regrets_ordered_win(player_move, oponent_move)
    def compute_regrets_ordered_win(self, winner_move, loser_move):
        if winner_move == ROCK: # loser played scissors
            self.regret_sum[SCISSORS] -= 1
            self.regret_sum[PAPER] -= 2
        if winner_move == SCISSORS: # loser played paper
            self.regret_sum[PAPER] -= 1
            self.regret_sum[ROCK] -= 2
        if winner_move == PAPER: # loser played rock
            self.regret_sum[ROCK] -= 1
            self.regret_sum[SCISSORS] -= 2
    def compute_regrets_ordered(self, loser_move, winner_move):
        if loser_move == ROCK: # the opponent must have played paper
            self.regret_sum[PAPER] += 1
            self.regret_sum[SCISSORS] += 2 
        elif loser_move == SCISSORS: # the opponent must have played rock
            self.regret_sum[ROCK] += 1
            self.regret_sum[PAPER] += 2
        elif loser_move == PAPER: # the opponent must have played scissors
            self.regret_sum[SCISSORS] += 1
            self.regret_sum[ROCK] += 2

    def compute_regrets(self, p1_move, p2_move):
        if p1_move == ROCK and p2_move == ROCK: 
            self.regret_sum[PAPER] += 2 # both oponents regret tying
        elif p1_move == PAPER and p2_move == PAPER:
            self.regret_sum[SCISSORS] += 2
        elif p1_move == SCISSORS and p2_move == SCISSORS:
            self.regret_sum[ROCK] += 2 
        else:
            if self.is_winner(p1_move, p2_move):
                self.compute_regrets_ordered(p1_move, p2_move) # the winner does not regret their move
            else:
                self.compute_regrets_ordered(p2_move, p1_move)
    
if __name__ == "__main__":
    rps = RockPaperScissorsTrainer()
    for i in range(NUM_EPISODES):
        strategy = rps.get_regret_strategy()
        p1_action = rps.get_action(strategy)
        p2_action = rps.get_action(strategy)
        rps.compute_regrets(p1_action, p2_action)
    print(rps.strategy_profile_sum / np.sum(rps.strategy_profile_sum))

    rps_opponent = RockPaperScissorsTrainer()
    opponent_strat = np.array([0.3, 0.4, 0.3])
    
    for i in range(NUM_EPISODES):
        strategy = rps_opponent.get_regret_strategy()
        player_action = rps_opponent.get_action(strategy)
        oponent_action = rps_opponent.get_action(opponent_strat)
        #print("oponent action: ", oponent_action, " our action: ", player_action)
        rps_opponent.compute_regrets_one_sided(player_action, oponent_action)
    print(rps_opponent.strategy_profile_sum / np.sum(rps_opponent.strategy_profile_sum))
    