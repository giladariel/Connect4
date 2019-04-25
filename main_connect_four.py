from connent_four import *
from human_player import *
from connoctfour_agent import *
from agent_tree import *

episodes = 300

game = ConnectFour(ConnectFourAgent, TreeAgent, 0.8, 0.8)
statistics = game.play_multiple_games(episodes, learn=False)
print('0.8, 0.8', statistics)

game = ConnectFour(ConnectFourAgent, HumanPlayer, 0.8, 0.8)
statistics = game.play_multiple_games(episodes, learn=False)
print('0.8, 0.8', statistics)
