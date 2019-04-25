import numpy as np
import random
import csv
import os


class TicRLAgent:

    def __init__(self, tag, exploration_factor=1):
        self.tag = tag
        self.exp_factor = exploration_factor
        self.prev_state = np.zeros(9)
        self.prev_move = -1
        self.state = None
        self.move = None
        self.count_memory = 0
        self.memory = {}
        self.values = {}
        self.load_values()
        self.cache = []

        self.alpha = 0.5
        self.gamma = 1

    def choose_move(self, state, winner, learn):

        self.load_to_memory(self.prev_state, self.prev_move, state, self.ava_moves(state), self.reward(winner))

        if winner is not None:

            self.count_memory += 1

            self.prev_state = np.zeros(9)
            self.prev_move = -1

            if learn is True and self.count_memory == 2:
                self.count_memory = 0
                # Offline training
                self.learn(self.memory)
                self.memory = {}

                # Online training
                # self.learn(self.prev_state, self.prev_move, state, self.ava_moves(state),  -1, self.reward(winner))
            return None

        p = random.uniform(0, 1)

        if p < self.exp_factor:
            idx = self.choose_optimal_move(state)
        else:
            ava_moves = self.ava_moves(state)
            idx = random.choice(ava_moves)

        self.prev_state = state
        self.prev_move = idx

        return idx

    def choose_optimal_move(self, state):
        ava_moves = self.ava_moves(state)
        v = -100

        idx = []
        for move in ava_moves:
            value = self.calc_value(state, move)

            if value > v:
                v = value
                idx = [move]
            elif v == value:
                idx.append(move)

        idx = random.choice(idx)
        return idx

    def reward(self, winner):

        if winner is self.tag:
            reward = 100
        elif winner is None:
            reward = 0
        elif winner == 0:
            reward = 50
        else:
            reward = -100
        return reward

    def learn(self, memory):

        for k, v in memory.items():
            [prev_state, prev_move, state, ava_moves, reward] = v

            v_s = self.calc_value(prev_state, prev_move)

            v = []
            key = np.array_str(self.state_to_vector(prev_state, prev_move))
            for move in ava_moves:
                v.append(self.calc_value(state, move))

            if reward == 0:
                if len(v) > 0:
                    v_s_tag = self.gamma * np.max(v)
                else:
                    print('no moves!!!')
                    v_s_tag = 0

                self.values[key] = np.array(v_s + self.alpha * (reward + v_s_tag - v_s))
            else:
                self.cache.append(key)
                self.values[key] = reward

    def calc_value(self, state, move):

        sa = self.state_to_vector(state, move)
        if np.array_str(sa) in self.values.keys():
            return self.values[np.array_str(sa)]
        else:
            return 0

    def load_values(self):
        s = 'values' + str(self.tag) + '.csv'
        try:
            value_csv = csv.reader(open(s, 'r'))
            for row in value_csv:
                k, v = row
                self.values[k] = float(v)
        except:
            pass
        # print(self.values)

    def save_values(self):
        s = 'values' + str(self.tag) + '.csv'
        try:
            os.remove(s)
        except:
            pass
        a = csv.writer(open(s, 'a'))

        for v, k in self.values.items():
            a.writerow([v, k])

    @staticmethod
    def ava_moves(state):
        moves = [s for s, v in enumerate(np.nditer(state)) if v == 0]
        return moves

    @staticmethod
    def state_to_vector(state, move):

        a = np.zeros(9)
        if move != -1:
            a[move] = 1
        tensor = np.array((a, state))
        tensor = tensor.reshape((1, 18))

        return tensor

    def load_to_memory(self, prev_state, prev_move, state, ava_moves, reward):
        key = np.array_str(self.state_to_vector(prev_state, prev_move))
        if key not in self.memory.keys():
            self.memory[key] = [prev_state, prev_move, state, ava_moves, reward]