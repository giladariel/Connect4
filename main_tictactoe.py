from tic_tac_toe import *
from tictactoe_agent import *

game = TicTacToe(TicTacToeAgent, TicTacToeAgent, 0.8, 0.8)
statistics = game.play_multiple_games(1000, learn=False)
print('0.8, 0.8', statistics)
