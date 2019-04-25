from abc import abstractmethod
import random
import numpy as np
import pickle
from model import *


class Agent:

    def __init__(self, tag, exploration_factor=1):
        self.tag = tag
        self.exp_factor = exploration_factor
        self.prev_state = np.zeros((6, 7))
        self.prev_move = -1
        self.state = None
        self.move = None
        self.print_value = False
        self.model = Model(self.tag)
        self.memory = []
        self.count_memory = 0
        self.winner_flag = False

    def choose_move(self, state, winner, learn):

        self.load_to_memory(self.prev_state, self.prev_move, state, self.ava_moves(state), self.reward(winner))

        if winner is not None:

            self.count_memory += 1

            self.prev_state = np.zeros((6, 7))
            self.prev_move = -1

            if learn is True and self.count_memory == 1000:
                self.count_memory = 0
                # Offline training
                self.model.learn_batch(self.memory)
                self.memory = []
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
        v = -float('Inf')
        v_list = []
        idx = []
        for move in ava_moves:
            value = self.model.calc_value(state, move)
            v_list.append(round(float(value), 5))

            if value > v:
                v = value
                idx = [move]
            elif v == value:
                idx.append(move)

        idx = random.choice(idx)
        return idx

    def game_winner(self, state):
        winner = None
        for i in range(len(state[:,0])-3):
            for j in range(len(state[0, :])-3):
                winner = self.square_winner(state[i:i+4, j:j+4])
                if winner is not None:
                    # print('winner is:', self.winner)
                    break
            if winner is not None:
                # print('winner is:', self.winner)
                break

        if np.min(np.abs(state[0, :])) != 0:
            winner = 0
            # print('no winner')

        return winner

    @staticmethod
    def square_winner(square):
        s = np.append([np.sum(square, axis=0), np.sum(square, axis=1).T],
                      [np.trace(square), np.flip(square,axis=1).trace()])
        if np.max(s) == 4:
            winner = 1
        elif np.min(s) == -4:
            winner = 2
        else:
            winner = None
        return winner

    @staticmethod
    def make_state_from_move(state, move, player):
        if move is None:
            return state

        state = np.array(state)
        if player == 1:
            tag = 1
        else:
            tag = -1

        if len(np.where(state[:, move] == 0)[0]) == 0:
            print(state)
        idy = np.where(state[:, move] == 0)[0][-1]
        state = np.array(state)
        state[idy, move] = tag

        return state

    def reward(self, winner):

        if winner is self.tag:
            reward = 1
        elif winner is None:
            reward = 0
        elif winner == 0:
            reward = 0.5
        else:
            reward = -1
        return reward

    def learn(self, prev_state, prev_move, state, ava_moves, move, reward):

        if prev_move != -1:

            target = self.model.calc_target(prev_state, prev_move, state, ava_moves, reward)
            # print(target)
            self.model.train_model(prev_state, prev_move, target, 1)

    @abstractmethod
    def ava_moves(self, state):
        pass

    def load_to_memory(self, prev_state, prev_move, state, ava_moves, reward):
        self.memory.append([prev_state, prev_move, state, ava_moves, reward])

    def save_memory(self):
        is_file_ = True
        count = 1
        s = ''
        while is_file_:
            s = 'data4/value_list_' + str(self.tag) + '_' + str(count) + '.pkl'
            if Path(s).is_file():
                is_file_ = True
                count = count + 1
            else:
                is_file_ = False

        with open(s, 'wb') as output:
            pickle.dump(self.memory, output)
