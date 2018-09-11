# ANN = Artificial Neural Networks

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

from sklearn.datasets import fetch_mldata
import numpy as np
import helper_functions as hf

def create_ann():
    ann = Sequential()
    ann.add(Dense(70, input_dim=784))
    ann.add(Activation('relu'))
    ann.add(Dense(50))
    ann.add(Activation('tanh'))
    ann.add(Dense(10))
    ann.add(Activation('softmax'))

    return ann

def train_ann(ann, x_train, y_train):
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    ann.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    ann.fit(x_train, y_train, epochs=20, batch_size=400, verbose=1)

    return ann

def initialize_ann():
    numbers = fetch_mldata('MNIST original', data_home='mnist')

    data = numbers.data / 255.0
    labels = numbers.target.astype('int')

    output = hf.convert_output(labels, 10)

    ann = create_ann()
    ann = train_ann(ann, data, output)

    ann.save('ann.obj')

# initialize_ann()