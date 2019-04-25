from model import *
import numpy as np
import keras.layers as Kl
import keras.models as Km
from keras import optimizers


class ConnectFourModel(Model):

    def __init__(self, tag):
        super().__init__(tag)
        pass

    def create_model(self):
        print('new model')

        model = Km.Sequential()
        model.add(Kl.Conv2D(20, (5, 5), padding='same', input_shape=(7, 7, 1)))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(20, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(20, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))

        model.add(Kl.Flatten(input_shape=(7, 7, 1)))
        model.add(Kl.Dense(49))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(7))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(7))
        model.add(Kl.LeakyReLU(alpha=0.3))

        model.add(Kl.Dense(1, activation='linear'))
        opt = optimizers.adam()
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

        model.summary()

        return model

    def state_to_tensor(self, state, move):

        vec = np.zeros((1, 7))
        if move != -1:
            vec[0, move] = 1
        tensor = np.append(vec, state, axis=0)
        tensor = tensor.reshape((1, 7, 7, 1))

        return tensor

