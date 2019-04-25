from abc import abstractmethod
import os
from pathlib import Path
import keras.models as Km
import keras as K
import numpy as np
import time


class Model:

    def __init__(self, tag):
        self.tag = tag
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 1
        self.model = self.load_model()

    def load_model(self):
        if self.tag == 1:
            tag = '_first'
        else:
            tag = '_second'
        s = 'model_values' + tag + '.h5'
        model_file = Path(s)

        if model_file.is_file():
            # print('load model')
            model = Km.load_model(s)
            # print('load model: ' + s)
        else:
            model = self.create_model()
        return model

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def state_to_tensor(self, state, move):
        pass

    def calc_value(self, state, move):
        tensor = self.state_to_tensor(state, move)
        value = self.model.predict(tensor)
        # K.backend.clear_session()
        return value

    def calc_target(self, prev_state, prev_move, state, ava_moves, reward):

        v_s = self.calc_value(prev_state, prev_move)

        v = []
        for move in ava_moves:
            v.append(self.calc_value(state, move))

        if reward == 0:
            if len(v) > 0:
                v_s_tag = self.gamma * np.max(v)
            else:
                print('no moves!!!')
                v_s_tag = 0
            target = np.array(v_s + self.alpha * (reward + v_s_tag - v_s))
        else:
            # v_s_tag = 0
            target = reward

        # target = np.array(v_s + self.alpha * (reward + v_s_tag - v_s))

        # if self.tag == 1:
        #     print('learn general')
        #     print(prev_state, prev_move, state, ava_moves, reward)
            # print('target: ', target)

        return target

    def train_model(self, prev_state, prev_move, target, epochs):

        tensor = self.state_to_tensor(prev_state, prev_move)

        if target is not None:

            # if self.tag == 1:
            #     print('value before training:', self.model.predict(tensor))
            self.model.fit(tensor, target, epochs=epochs, verbose=0)
            # K.backend.clear_session()

            # if self.tag == 1:
            #     print('target:', target)
            #     print('value after training:', self.model.predict(tensor))

    def save_model(self):
        if self.tag == 1:
            tag = '_first'
        else:
            tag = '_second'
        s = 'model_values' + tag + '.h5'

        try:
            os.remove(s)
        except:
            pass

        self.model.save(s)

    def learn_batch(self, memory):
        print('start learning player', self.tag)
        print('data length:', len(memory))

        # build x_train
        ind = 0
        x_train = np.zeros((len(memory), 7, 7, 1))
        # x_train = np.zeros((len(memory), 2, 9))
        for v in memory:
            [prev_state, prev_move, _, _, _] = v
            sample = self.state_to_tensor(prev_state, prev_move)
            x_train[ind, :, :, :] = sample
            ind += 1

        # train with planning
        # for i in range(5):
        loss = 20
        count = 0
        while loss > 0.02:
            # tic()
            y_train = self.create_targets(memory)
            # toc()
            self.model.fit(x_train, y_train, epochs=5, batch_size=256, verbose=0)
            loss = self.model.evaluate(x_train, y_train, batch_size=256, verbose=0)[0]
            count += 1
            print('planning number:', count, 'loss', loss)

        loss = self.model.evaluate(x_train, y_train, batch_size=256, verbose=0)
        # print('player:', self.tag, loss, 'loops', count)

        self.save_model()

    def create_targets(self, memory):
        y_train_ = np.zeros((len(memory), 1))
        count_ = 0
        for v_ in memory:
            [prev_state_, prev_move_, state_, ava_moves_, reward_] = v_
            target = self.calc_target(prev_state_, prev_move_, state_, ava_moves_, reward_)
            y_train_[count_, :] = target
            count_ += 1

            # print('---------')
            # print('player', self.tag)
            # print('prev state', prev_state_)
            # print('prev move', prev_move_)
            # print('state', state_)
            # print('ava moves', ava_moves_)
            # print('reward', reward_)
            # print('target', target)
            #
            # value = self.calc_value(prev_state_, prev_move_)
            # print('value through net', value)
            # time.sleep(0.2)

        return y_train_
