from agent import *
from model_connectfour import *


class ConnectFourAgent(Agent):

    def __init__(self, tag, exploration_factor=1):

        super().__init__(tag, exploration_factor)
        self.model = ConnectFourModel(tag)
        self.prev_state = np.zeros((6, 7))

    def ava_moves(self, state):
        moves = np.where(state[0, :] == 0)[0]
        return moves
