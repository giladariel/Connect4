from agent_RL import *
from abc import abstractmethod
from agent_tree import *


class Game:

    def __init__(self, player1, player2, exp1=1, exp2=1, tag1=1, tag2=2):

        self.players = {1: player1(tag1, exploration_factor=exp1),
                        2: player2(tag2, exploration_factor=exp2)}

        self.state, self.winner, self.turn = self.init_game()
        self.memory = {}

    def play_game(self, learn=False):

        move_count = 0

        while self.winner is None:
            move = self.play_move(learn)

            self.state = self.make_state_from_move(move)
            self.game_winner()

            self.next_player()
            move_count += 1

        self.play_move(learn)
        self.next_player()
        self.play_move(learn)
        self.next_player()

        return self.winner, move_count

    def play_move(self, learn):
        player = self.players[self.turn]
        move = player.choose_move(self.state, self.winner, learn)
        return move

    def play_multiple_games(self, episodes, learn):
        statistics = {1: 0, 2: 0, 0: 0, 'move_count': 0}
        move_count_total = []
        for i in range(episodes):
            winner, move_count = self.play_game(learn)
            move_count_total.append(move_count)
            statistics[winner] = statistics[winner] + 1

            self.state, self.winner, self.turn = self.init_game()

        if isinstance(self.players[1], TicRLAgent):
            self.players[1].save_values()
        if isinstance(self.players[2], TicRLAgent):
            self.players[2].save_values()

        if learn is True and isinstance(self.players[1], TreeAgent):
            self.players[1].save_tree()
        if learn is True and isinstance(self.players[2], TreeAgent):
            self.players[2].save_tree()

        statistics['move_count'] = np.mean(move_count_total)
        return statistics

    @abstractmethod
    def init_game(self):
        pass

    @abstractmethod
    def make_state_from_move(self, move):
        pass

    @abstractmethod
    def next_player(self):
        pass

    @abstractmethod
    def game_winner(self):
        pass

    @abstractmethod
    def print_game(self):
        pass
