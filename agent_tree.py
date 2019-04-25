import numpy as np
import os

from connect_tree import *
from pathlib import Path


class TreeAgent:

    def __init__(self, tag, exploration_factor):
        self.tag = tag
        self.c = np.sqrt(2)/10
        self.exp_factor = exploration_factor
        self.play_tree = self.load_tree(tag)
        self.expand_flag = True
        global t
        global con_tree

    @staticmethod
    def ava_moves(state):
        moves = np.where(state[0, :] == 0)[0]
        # print(moves)
        return moves

    def choose_move2(self, state, winner, learn):
        # print('choose move')

        global con_tree

        if winner is not None:
            # update tree
            print('winner:', winner, 'tag:', self.tag)
            print('tree depth:', con_tree.depth)
            while con_tree.prev is not None and winner == self.tag:
                if winner == self.tag:
                    con_tree.num_win += 1
                con_tree = con_tree.prev
                con_tree.num_visit += 1
            # print('winner is:', winner)
            return

        ava_moves = self.ava_moves(state)

        if learn is False and con_tree is None:
            idx = random.choice(ava_moves)
            return idx

        if self.exp_factor == 0:
            idx = random.choice(ava_moves)
            con_tree = getattr(con_tree, 'm' + str(idx + 1))
            return idx

        mct = 0
        idx = []

        for move in ava_moves:

            if getattr(con_tree, 'm' + str(move+1)) is None:
                idx.append(move)
        if len(idx) > 0:
            idx = random.choice(idx)
            new_state = self.make_state_from_move(state, idx)

            if self.tag == 1:
                tag = 2
            else:
                tag = 1

            if learn is True:
                con_tree = con_tree.expand(idx+1, new_state, tag)

            return idx

        for move in ava_moves:

            num_vis = getattr(getattr(con_tree, 'm' + str(move + 1)), 'num_visit')
            num_w = getattr(getattr(con_tree, 'm' + str(move + 1)), 'num_win')

            total_num = con_tree.num_visit

            value = num_w/num_vis + self.c * np.sqrt(np.log(total_num / num_vis))

            if value >= mct:
                mct = value
                idx = move

        con_tree = getattr(con_tree, 'm' + str(idx + 1))

        return idx

    def choose_move(self, state, winner, learn):

        self.expand_opp_move(state, learn)
        if self.exp_factor == 0 and winner is None:
            moves = self.ava_moves(state)
            return random.choice(moves)

        # expand tree to opponent move

        if winner is not None:
            self.back_prop_tree(winner)
            if learn is True:
                self.expand_flag = True
            return

        # check if all leaf init
        ava_moves = self.ava_moves(state)
        idx = []

        for move in ava_moves:
            if getattr(self.play_tree, 'm' + str(move+1)) is None:
                idx.append(move)

        if len(idx) > 0:
            leaf_init = False
        else:
            leaf_init = True

        # if true - pick Bandit
        if leaf_init is True:
            move = self.pick_bandit(ava_moves)
            self.play_tree = getattr(self.play_tree, 'm' + str(move + 1))
            return move

        # if false - there is at least 1 None
        else:
            # if learn false pick random
            if learn is False:
                return random.choice(ava_moves)
            # if learn true pick random from none and expand
            else:
                move = random.choice(idx)
                new_state = self.make_state_from_move(state, move)

                if self.tag == 1:
                    tag = 2
                else:
                    tag = 1

                if self.expand_flag is True:
                    self.play_tree = self.play_tree.expand(move + 1, new_state, tag)

            self.expand_flag = False
            return move

    def expand_opp_move(self, state, learn):

        if self.exp_factor == 0 or self.expand_flag is False:
            return

        prev_state = self.play_tree.state
        diff = prev_state - state
        _, idx = np.nonzero(diff)

        if len(idx) == 0:
            return

        idx = idx[0]

        if self.tag == 1:
            opp_tag = 2
        else:
            opp_tag = 1

        if getattr(self.play_tree, 'm' + str(idx+1)) is None:
            if learn is False:
                pass
            else:
                self.play_tree = self.play_tree.expand(idx+1, state, opp_tag)
        else:
            self.play_tree = getattr(self.play_tree, 'm' + str(idx + 1))

    def back_prop_tree(self, winner):
        while self.play_tree.prev is not None:
            if winner == self.tag:
                self.play_tree.num_win += 1

            self.play_tree.num_visit += 1

            self.play_tree = self.play_tree.prev

        if winner == self.tag:
            self.play_tree.num_win += 1
        self.play_tree.num_visit += 1

        return

    def pick_bandit(self, ava_moves):

        mct = 0
        idx = []

        for move in ava_moves:
            num_vis = getattr(getattr(self.play_tree, 'm' + str(move + 1)), 'num_visit')
            num_w = getattr(getattr(self.play_tree, 'm' + str(move + 1)), 'num_win')

            total_num = self.play_tree.num_visit

            value = num_w / num_vis + self.c * np.sqrt(np.log(total_num / num_vis))

            if value > mct:
                mct = value
                idx = [move]
            elif value == mct:
                idx.append(move)

        return random.choice(idx)

    def make_state_from_move(self, state, move):
        if self.tag == 1:
            tag = 1
        else:
            tag = -1
        idy = np.where(state[:, move] == 0)[0][-1]
        new_state = np.array(state)
        new_state[idy, move] = tag
        return new_state

    def save_tree(self):
        s = 'Trees/Tree' + str(self.tag) + '.pkl'

        try:
            os.remove(s)
        except:
            pass

        with open(s, 'wb') as output:
            pickle.dump(self.play_tree, output)

    @staticmethod
    def load_tree(tag):
        s = 'Trees/Tree' + str(tag) + '.pkl'
        print(s)
        tree_file = Path(s)
        if tree_file.is_file():
            print('load tree tag:', tag)
            with open(s, 'rb') as input_:
                tr = pickle.load(input_)
                return tr
        else:
            print('new tree tag:', tag)
            return ConnectTree(np.zeros((6, 7)), 1)


if __name__ == '__main__':

    tr = TreeAgent(1, 1)
    for i in range(7):
        num_vis = getattr(getattr(tr.play_tree, 'm' + str(i + 1)), 'num_visit')
        num_w = getattr(getattr(tr.play_tree, 'm' + str(i + 1)), 'num_win')
        print('-----------')
        print('move:', i)
        print('visits', num_vis)
        print('wins', num_w)
        moves = tr.ava_moves(tr.play_tree.state)

