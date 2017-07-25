# coding: utf-8
""" Prediction with CNN
input: fisheye image
out: Generated Power
クロスバリデーションもする
とりあえずkeras
"""

# library
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as import pdb
import datetime as dt
import os
import sys
import seaborn as sns
matplotlib.use('Agg')

SAVE_dir = "./RESULT/CNN_keras/"
if not os.path.isdir(SAVE_dir):
    os.makedirs(SAVE_dir)

def CNN_model1(activation="relu", loss="mean_squared_error", optimizer="Adadelta"):
    """
    INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*2 -> [FC -> RELU]*2 -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(layer, height, width)))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def CNN_model2(activation="relu", loss="mean_squared_error", optimizer="Adadelta"):
    """
    INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(layer, height, width)))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation=activation))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def CNN_model3(activation="relu", loss="mean_squared_error", optimizer="Adadelta"):
    """
    INPUT -> [CONV -> RELU] -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(layer, height, width)))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model









































# end
