import copy
import random


class ConnectTree:

    def __init__(self, state, tag, depth=0):
        self.state = state
        self.m1 = None
        self.m2 = None
        self.m3 = None
        self.m4 = None
        self.m5 = None
        self.m6 = None
        self.m7 = None
        self.prev = None

        self.tag = tag
        self.num_visit = 0
        self.num_win = 0
        self.depth = depth

    def expand(self, move, state, tag):
        sub_tree = ConnectTree(state, tag, self.depth + 1)
        setattr(self, 'm' + str(move), sub_tree)
        sub_tree.prev = self
        return sub_tree

    def count_tree_states(self):
        count_tree = copy.deepcopy(self)
        while count_tree.prev is not None:
            count_tree = count_tree.prev

        count = self.count_from_root(count_tree) - 1

        return count

    def count_from_root(self, count_tree):
        count = 1
        if count_tree is not None:
            for i in range(7):
                if getattr(count_tree, 'm' + str(i + 1)) is not None:
                    count += self.count_from_root(getattr(count_tree, 'm' + str(i + 1)))

        return count

    def print_random_branch(self, tree):
        a = []
        for i in range(7):
            if getattr(tree, 'm' + str(i + 1)) is not None:
                a.append(i)
        if len(a) > 0:
            branch = random.choice(a)
            print(' -->', branch + 1, end='')
            tree = getattr(tree, 'm' + str(branch + 1))
            self.print_random_branch(tree)
        else:
            print(tree.state)
