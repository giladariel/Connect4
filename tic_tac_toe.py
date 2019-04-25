from game import *
import numpy as np


class TicTacToe(Game):

    def __init__(self, player1, player2, exp1=1, exp2=1, tag1=1, tag2=2):
        super().__init__(player1, player2, exp1, exp2, tag1, tag2)

    def init_game(self):
        return np.zeros(9), None, 1

    def make_state_from_move(self, move):
        if move is None:
            return self.state

        state = np.array(self.state)

        if self.turn == 1:
            tag = 1
        else:
            tag = -1

        state[move] = tag
        return state

    def next_player(self):
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1

    def game_winner(self):

        winner = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

        for line in winner:

            s = self.state[line[0]] + self.state[line[1]] + self.state[line[2]]
            if s == 3:
                self.winner = 1
                break
            elif s == -3:
                self.winner = 2
                break
            elif not any(s == 0 for s in np.nditer(self.state)):
                self.winner = 0

    def print_game(self):

        s = self.state

        print('    {} | {} | {}'.format(s[0], s[1], s[2]))
        print('  --------------')
        print('    {} | {} | {}'.format(s[3], s[4], s[5]))
        print('  --------------')
        print('    {} | {} | {}'.format(s[6], s[7], s[8]))
        print('  --------------')
        print('  --------------')
