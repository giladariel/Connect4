import numpy as np
import random
from agent import *
from model_tictactoe import *


class TicTacToeAgent(Agent):

    def __init__(self, tag, exploration_factor=1):

        super().__init__(tag, exploration_factor)
        self.model = TicTacToeModel(tag)

    def ava_moves(self, state):
        moves = [s for s, v in enumerate(np.nditer(state)) if v == 0]
        return moves
