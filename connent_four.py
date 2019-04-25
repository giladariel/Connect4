from game import *
import numpy as np


class ConnectFour(Game):

    def __init__(self, player1, player2, exp1=1, exp2=1, tag1=1, tag2=2):
        super().__init__(player1, player2, exp1, exp2, tag1, tag2)

    def init_game(self):
        return np.zeros((6, 7)), None, 1

    def make_state_from_move(self, move):
        if move is None:
            return self.state

        state = np.array(self.state)
        if self.turn == 1:
            tag = 1
        else:
            tag = -1

        if len(np.where(state[:, move] == 0)[0]) == 0:
            print(state)
        idy = np.where(state[:, move] == 0)[0][-1]
        state = np.array(state)
        state[idy, move] = tag

        return state

    def next_player(self):
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1

    def game_winner(self):
        for i in range(len(self.state[:,0])-3):
            for j in range(len(self.state[0, :])-3):
                self.square_winner(self.state[i:i+4, j:j+4])
                if self.winner is not None:
                    # print('winner is:', self.winner)
                    break
            if self.winner is not None:
                # print('winner is:', self.winner)
                break

        if np.min(np.abs(self.state[0, :])) != 0:
            self.winner = 0
            # print('no winner')

    def square_winner(self, square):
        s = np.append([np.sum(square, axis=0), np.sum(square, axis=1).T],
                      [np.trace(square), np.flip(square,axis=1).trace()])
        if np.max(s) == 4:
            self.winner = 1
        elif np.min(s) == -4:
            self.winner = 2
        else:
            self.winner = None
        return self.winner

    def print_game(self):

        print('  --------------')
        print(self.state)
        print('  --------------')
