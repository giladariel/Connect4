from model import *
import numpy as np
import keras.layers as Kl
import keras.models as Km
import keras.optimizers as Ko


class TicTacToeModel(Model):

    def __init__(self, tag):
        super().__init__(tag)
        pass

    def create_model(self):
        print('new model')

        model = Km.Sequential()
        model.add(Kl.Flatten(input_shape=(2, 9)))
        model.add(Kl.Dense(18))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(18))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(18))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(18))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(18))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(18))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(18))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(9))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(1, activation='linear'))

        # adam = Ko.Adam(lr=0.001)
        model.compile(optimizer='Adam', loss='mean_absolute_error', metrics=['accuracy'])

        model.summary()

        return model

    def state_to_tensor(self, state, move):

        a = np.zeros(9)
        if move != -1:
            a[move] = 1
        tensor = np.array((a, state))
        tensor = tensor.reshape((1, 2, 9))

        return tensor
