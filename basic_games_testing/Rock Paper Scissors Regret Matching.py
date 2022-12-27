import numpy as np
import random
import matplotlib.pyplot as plt

NUM_EPISODES = 10000
NUM_ACTIONS = 3
ROCK = 0
PAPER = 1
SCISSORS = 2

class RockPaperScissorsTrainer():
    def __init__(self):
        self.cummulative_strategy_profile = np.zeros(NUM_ACTIONS)
        self.cummulative_regrets = np.zeros(NUM_ACTIONS)
        self.num_iters = 0
        self.rock_regrets = []
        self.paper_regrets = []
        self.scissors_regrets = []
    def get_regret_strategy(self):
        regret_sum = np.sum(self.cummulative_regrets[self.cummulative_regrets > 0]) # only use positive regrets
        if regret_sum <= 0:
            strategy = np.ones(NUM_ACTIONS) / NUM_ACTIONS
        else:
            strategy = np.zeros(NUM_ACTIONS)
            for i in range(NUM_ACTIONS):
                if(self.cummulative_regrets[i] > 0):
                    strategy[i] = self.cummulative_regrets[i] / regret_sum
        self.cummulative_strategy_profile += strategy
        self.num_iters += 1
        return strategy
    def get_action(self, strategy):
        return np.random.choice([ROCK, PAPER, SCISSORS], p=strategy)
    def update_regrets_one_sided(self, player_move, oponent_move):
        current_move_regrets = self.compute_regrets(player_move, oponent_move)
        self.cummulative_regrets += current_move_regrets # add the current regrets to the cummulative regrets
    def compute_action_value_array(self, opponent_action):
        action_array = np.zeros(NUM_ACTIONS)
        action_array[opponent_action] = 0 # if we play opponents action the utility is zero
        action_array[((opponent_action + 1) % 3)] = 1
        action_array[((opponent_action - 1) % 3)] = -1
        return action_array
    def compute_regrets(self, player_action, opponent_action):
        action_array = self.compute_action_value_array(opponent_action)
        regrets = np.zeros(NUM_ACTIONS)
        for i in range(NUM_ACTIONS):
            regrets[i] = action_array[i] - action_array[player_action]
        return regrets
if __name__ == "__main__":
    
    # start two agents with a random strategy and check if they both converge to nash equilibrium
    rps_one = RockPaperScissorsTrainer()
    rps_two = RockPaperScissorsTrainer()
    random_strat_one = np.random.rand(3) 
    random_strat_two = np.random.rand(3)
    random_strat_one /= np.sum(random_strat_one)
    random_strat_two /= np.sum(random_strat_two)
    for i in range(NUM_EPISODES):
        strategy_one = rps_one.get_regret_strategy()
        strategy_two = rps_two.get_regret_strategy()
        if i == 0:
            strategy_one = random_strat_one
            strategy_two = random_strat_two
        p1_action = rps_one.get_action(strategy_one)
        p2_action = rps_two.get_action(strategy_two)
        rps_one.update_regrets_one_sided(p1_action, p2_action)
        rps_two.update_regrets_one_sided(p2_action, p1_action)

    print(rps_one.cummulative_strategy_profile / np.sum(rps_one.cummulative_strategy_profile))
    print(rps_two.cummulative_strategy_profile / np.sum(rps_two.cummulative_strategy_profile))


    #check response to single agent with fixed policy (picks scissors all the time, will do math to make sure this is optimal later)
    rps_opponent = RockPaperScissorsTrainer()
    opponent_strat = np.array([0.3, 0.4, 0.3])
    
    for i in range(NUM_EPISODES):
        strategy = rps_opponent.get_regret_strategy()
        player_action = rps_opponent.get_action(strategy)
        oponent_action = rps_opponent.get_action(opponent_strat)
        rps_opponent.update_regrets_one_sided(player_action, oponent_action)

    print(rps_opponent.cummulative_strategy_profile / np.sum(rps_opponent.cummulative_strategy_profile))
    
    #match nash eq opponent
    #this does not work, not sure what is happening. 
    rps_opponent_nash = RockPaperScissorsTrainer()
    opponent_strat = np.array([0.333333333333333333333333333333, 0.333333333333333333333333333333, 0.333333333333333333333333333334])
    
    for i in range(NUM_EPISODES):
        strategy = rps_opponent_nash.get_regret_strategy()
        player_action = rps_opponent_nash.get_action(strategy)
        oponent_action = rps_opponent_nash.get_action(opponent_strat)
        rps_opponent_nash.update_regrets_one_sided(player_action, oponent_action)

    print(rps_opponent_nash.cummulative_strategy_profile / np.sum(rps_opponent_nash.cummulative_strategy_profile))