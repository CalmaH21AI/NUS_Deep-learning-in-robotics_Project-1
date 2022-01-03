import numpy as np
import random
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
# "tabulate" is used to show Q-table in function "show_Q"

class fl:
    def __init__(self, grid, start, wins, lr, dr, eps, fails=None, hole_rate=None):
        # input
        self.rows = grid[0]
        self.cols = grid[1]
        self.start = start
        self.wins = wins
        self.fails = fails
        self.lr = lr # learning rate
        self.dr = dr # discount rate
        self.eps = eps # epsilon
        self.hole_rate = hole_rate # proportion of holes
        # learning outcomes
        self.Q = [[[0]*4 for j in range(self.rows)] for i in range(self.cols)] # Q-table
        self.epi_values = [] # a list of average state-action values for line plot
        self.episodes = [] # a list of route lists for all episodes
        self.epi_num = 0 # the number of episodes
        self.win_num = 0 # the number of win episodes
        self.win_index = None # the first index of win episodes
        self.optimal_num = 0 # the number of optimal episodes
        self.optimal_step = None # the step number of the optimal episode
        self.optimal_route = [] # the route of the optimal episode
        self.optimal_index = None # the first index of optimal episodes
        self.epi_infos = [] # [epi_num, step_num, is_win]
        self.algo = None # "MC", "SS", or "QL"
    
    def get_grid(self):
        if self.fails != None:
            pass
        else:
            # method: generate random win-route and then randomly scatter holes
            too_long = True # path might cover more than (1-hole_rate) of the map
            state = self.start
            # generate a win path
            while too_long:
                is_end = False
                route = []
                route.append(self.start)
                while not is_end:
                    action = np.random.choice([0,1,2,3])
                    if action == 0:
                        state_ = (state[0] - 1, state[1])
                    elif action == 1:
                        state_ = (state[0] + 1, state[1])
                    elif action == 2:
                        state_ = (state[0], state[1] - 1)
                    else: # action == 3
                        state_ = (state[0], state[1] + 1)
                    if state_[0]<0 or state_[0]>self.rows-1 or state_[1]<0 or state_[1]>self.cols-1 or state == state_:
                        state_ = state
                    else:
                        route.append(state_)
                    state = state_
                    if state in self.wins:
                        is_end = True
                # cut the repeated nodes
                path_dic = {} # use dictionary to record connectivity (key and value is connected)
                for i in range(len(route)-1):
                    path_dic[route[i]] = route[i+1]
                path = []
                key = self.start
                path.append(key)
                while True:
                    path.append(path_dic[key])
                    key = path_dic[key]
                    if key in self.wins:
                        break
                if len(path) < self.rows*self.cols*(1.0-self.hole_rate):
                    too_long = False
            # randomly scatter holes
            blanks = []
            for i in range(self.rows):
                for j in range(self.cols):
                    if (i,j) not in path and (i,j) != self.start and (i,j) not in self.wins:
                        blanks.append((i,j))
            holes = []
            holes_index = np.random.choice(len(blanks), size=int(self.rows*self.cols*self.hole_rate), replace=False)
            for i in holes_index:
                holes.append(blanks[i])
            self.fails = holes
            
    def show_grid(self, state=None):
        if state == None:
            state = self.start
        for i in range(self.cols):
            print("-",end="")
        print("")
        for i in range(self.rows):
            for j in range(self.cols):
                if (i,j) in self.fails:
                    print("8", end="")
                elif (i,j) in self.wins:
                    print("1", end="")
                elif (i,j) == state:
                    print("*", end="")
                else:
                    print("O", end="")
            print("")
        for i in range(self.cols):
            print("-",end="")
        print("")
    
    def s2p(self, state):
        position = state[0]*self.cols+state[1]+1
        return position
    
    def choose_action(self, state):
        # find the index(es) with max value
        av = [] # action-value pair
        # find action(s) with maximum value(s) excluding Nones
        for (i,j) in enumerate(self.Q[state[0]][state[1]]):
            if j != None:
                av.append([i,j])
        max_value = max(j for [i,j] in av)
        max_indices = []
        for (i,j) in enumerate(self.Q[state[0]][state[1]]):
            if j == max_value:
                max_indices.append(i)
        # randomly select one action with max value
        max_index = random.choice(max_indices)
        # choose action with probabilities (ε-greedy)
        actions = []
        probs = []
        for [i,j] in av:
            if i == max_index:
                probs.append(1-self.eps+self.eps/len(av))
            else:
                probs.append(self.eps/len(av))
            actions.append(i)
        action = random.choices(actions, weights=probs, k=1)
        return action[0]
    
    def choose_max(self, state): # for QL
        # find the action with max value in the next state
        max_value = max(i for i in self.Q[state[0]][state[1]] if i != None)
        max_indices = []
        for (i,j) in enumerate(self.Q[state[0]][state[1]]):
            if j == max_value:
                max_indices.append(i)
        # randomly select one action with max value
        max_index = random.choice(max_indices)
        return max_index
    
    def take_action(self, state, action):
        # action: up(0) down(1) left(2) right(3)
        if action == 0:
            next_state = (state[0] - 1, state[1])
        elif action == 1:
            next_state = (state[0] + 1, state[1])
        elif action == 2:
            next_state = (state[0], state[1] - 1)
        elif action == 3:
            next_state = (state[0], state[1] + 1)
        else:
            print("Wrong action.")
        if next_state[0]<0 or next_state[0]>self.rows-1 or next_state[1]<0 or next_state[1]>self.cols-1:
            next_state = state
            self.update_values(state, action, None)
        return next_state
    
    def update_values(self, state, action, value): # for TD algorithems
        self.Q[state[0]][state[1]][action] = value
    
    def get_reward(self, state):
        if state in self.wins:
            return 1
        elif state in self.fails:
            return -1
        else:
            return 0
        
    def get_value(self, state, action, reward, next_state, next_action): # for TD algorithems
        q0 = self.Q[state[0]][state[1]][action]
        q1 = self.Q[next_state[0]][next_state[1]][next_action]
        if q0 == None:
            return None
        else:
            q0_ = q0+self.lr*(reward+self.dr*q1-q0)
            return q0_
        
    def get_action_value(self, state, action):
        v = self.Q[state[0]][state[1]][action]
        if v == None:
            return 0
        else:
            return v
        
    def show_Q(self):
        row = self.rows*self.cols
        value = np.resize(self.Q,(row,4))
        table = [[]]*row
        for i in range(row):
            table[i].append(i+1)
            for j in range(len(value[i])):
                table[i].append(value[i][j])
        for i in range(len(table)):
            np.insert(table[i], 0, i+1)
        header = ["State","UP","DOWN","LEFT","RIGHT"]
        tab = np.resize(table,(row,5))
        print(tabulate(tab, headers=header))
    
    def MC(self, iteration):
        self.algo = "MC"
        for i in range(iteration):
            state = self.start
            is_end = False
            is_win = False
            step_num = 0
            sar = [] # a list of [(state),action]
            episode = [self.s2p(state)]
            epi_value = [] # averaged state-action values in an episode
            while not is_end:
                clear_output(wait=True)
                print("Episode:",self.epi_num)
                print("Step:",step_num)
                # self.show_grid(state)
                step_num += 1
                action = self.choose_action(state) ########
                epi_value.append(self.get_action_value(state, action))
                next_state = self.take_action(state, action)
                next_action = self.choose_action(next_state)
                reward = self.get_reward(next_state)
                sar.append([state, action, reward])
                state = next_state
                episode.append(self.s2p(state))
                if state in self.wins or state in self.fails:
                    is_end = True
            if state in self.wins:
                is_win = True
                self.win_num += 1
                if self.win_index == None:
                    self.win_index = self.epi_num
                if self.optimal_step == None or step_num < self.optimal_step:
                    self.optimal_step = step_num
                    self.optimal_route = episode
                    self.optimal_num = 1
                    self.optimal_index = self.epi_num
                elif step_num == self.optimal_step:
                    self.optimal_num += 1
                else:
                    pass
            self.episodes.append(episode)
            self.epi_infos.append([self.epi_num, step_num, is_win])
            self.epi_num += 1
            self.epi_values.append(sum(epi_value)/len(epi_value))
            # update Q values
            for j in range(len(sar)):
                if j != 0:
                    sar[-j-1][2] += self.dr*sar[-j][2]
            returns = []
            visited = []
            for j in sar:
                if (j[0][0],j[0][1]) not in visited:
                    returns.append(j)
                visited.append((j[0][0],j[0][1]))
            for j in returns:
                if self.Q[j[0][0]][j[0][1]][j[1]] != None:
                    self.Q[j[0][0]][j[0][1]][j[1]] = (1-self.lr)*self.Q[j[0][0]][j[0][1]][j[1]]+self.lr*j[2]
        
    def SS(self, iteration):
        self.algo = "SS"
        for i in range(iteration):
            state = self.start
            is_end = False
            is_win = False
            step_num = 0
            action = self.choose_action(state)
            episode = [self.s2p(state)]
            epi_value = []
            while not is_end:
                clear_output(wait=True)
                print("Episode:",self.epi_num)
                print("Step:",step_num)
                # self.show_grid(state)
                step_num += 1
                next_state = self.take_action(state, action)
                reward = self.get_reward(next_state)
                next_action = self.choose_action(next_state)
                value = self.get_value(state, action, reward, next_state, next_action)
                epi_value.append(self.get_action_value(state, action))
                self.update_values(state, action, value)
                last_action = action
                action = next_action
                state = next_state
                episode.append(self.s2p(state))
                if state in self.wins or state in self.fails:
                    is_end = True
            if state in self.wins:
                is_win = True
                self.win_num += 1
                if self.win_index == None:
                    self.win_index = self.epi_num
                if self.optimal_step == None or step_num < self.optimal_step:
                    self.optimal_step = step_num
                    self.optimal_route = episode
                    self.optimal_num = 1
                    self.optimal_index = self.epi_num
                elif step_num == self.optimal_step:
                    self.optimal_num += 1
                else:
                    pass
            self.episodes.append(episode)
            self.epi_infos.append([self.epi_num, step_num, is_win])
            self.epi_num += 1
            self.epi_values.append(sum(epi_value)/len(epi_value))
        
    def QL(self, iteration):
        self.algo = "QL"
        for i in range(iteration):
            state = self.start
            is_end = False
            is_win = False
            step_num = 0
            action = self.choose_action(state)
            episode = [self.s2p(state)]
            epi_value = []
            while not is_end:
                clear_output(wait=True)
                print("Episode:",self.epi_num)
                print("Step:",step_num)
                # self.show_grid(state)
                step_num += 1
                next_state = self.take_action(state, action)
                reward = self.get_reward(next_state)
                next_action = self.choose_action(next_state)
                
                max_action = self.choose_max(next_state) # for value to be updated
                value = self.get_value(state, action, reward, next_state, max_action) # max <- next
                
                epi_value.append(self.get_action_value(state, action))
                self.update_values(state, action, value)
                last_action = action
                action = next_action
                state = next_state
                episode.append(self.s2p(state))
                if state in self.wins or state in self.fails:
                    is_end = True
            if state in self.wins:
                is_win = True
                self.win_num += 1
                if self.win_index == None:
                    self.win_index = self.epi_num
                if self.optimal_step == None or step_num < self.optimal_step:
                    self.optimal_step = step_num
                    self.optimal_route = episode
                    self.optimal_num = 1
                    self.optimal_index = self.epi_num
                elif step_num == self.optimal_step:
                    self.optimal_num += 1
                else:
                    pass
            self.episodes.append(episode)
            self.epi_infos.append([self.epi_num, step_num, is_win])
            self.epi_num += 1
            self.epi_values.append(sum(epi_value)/len(epi_value))
    
    def learning(self, algo, iteration):
        if algo == "MC":
            self.MC(iteration)
        elif algo == "SS":
            self.SS(iteration)
        elif algo == "QL":
            self.QL(iteration)
        else:
            print("Please input 'MC', 'SS', or 'QL'.")
        
    def reset(self):
        self.Q = [[[0]*4 for j in range(self.rows)] for i in range(self.cols)]
        self.avg_rewards = 0
        self.episodes = []
        self.epi_num = 0
        self.win_num = 0
        self.win_index = None
        self.optimal_num = 0
        self.optimal_step = None
        self.optimal_route = []
        self.optimal_index = None
        self.epi_infos = []
        self.algo = None
        
# plotting

def lplot(mc=None, ss=None, ql=None):
    plt.plot(mc, "o", markersize=1, label="MC")
    plt.plot(ss, "o", markersize=1, label="SARSA")
    plt.plot(ql, "o", markersize=1, label="QLearning")
    plt.legend()
    plt.xlabel('The number of episode')
    plt.ylabel('Average state-action reward')
    plt.show()

def pplot(data, algo):
        e = [] # episode_num
        s = [] # step_num
        w = [] # is_win
        for i,j,k in data:
            e.append(i)
            s.append(j)
            w.append(k)
        df = pd.DataFrame({"Episode":e, "Step":s, "Win":w})
        groups = df.groupby("Win")
        for name, group in groups:
            if name == True:
                label = "Win"
            else:
                label = "Fail"
            plt.plot(group.Episode, group.Step, marker='o', linestyle='', markersize=2, label=label)
        plt.legend(loc='upper right', shadow=True, fontsize='xx-large')
        plt.xlabel("Episode number")
        plt.ylabel("Step number")
        plt.title(label=algo)
        plt.show()

import seaborn as sns

def show_heatmap(Q, algo):
    data = np.array(Q)
    (x,y,z) = data.shape
    values = np.zeros([x,y])
    actions = np.zeros([x,y])
    hole = np.zeros(z)
    for i in range(x):
        for j in range(y):
            if np.array_equal(data[i][j], hole):
                actions[i][j] = None
            else:
                av = {}
                for a,v in enumerate(data[i][j]):
                    av[a]=v
                value = max(av[k] for k in av if av[k] != None) # find max value
                # find the index of the max value
                for key, val in av.items():
                    if val == value:
                        actions[i][j] = key
    sns.heatmap(actions, annot=True, linewidths=1)
    plt.title("{}\n0-UP | 1-DOWN | 2-LEFT | 3-RIGHT".format(algo))
    plt.show()