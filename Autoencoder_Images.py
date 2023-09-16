import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers import SGD
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#scale images
X_train = X_train/255
X_test = X_test/255

#Build stacked autoencoder that reduces the dimensionality of the dataset
encoder = Sequential()
encoder.add(Flatten(input_shape=[28,28]))
encoder.add(Dense(inputs=400, activation='relu'))
encoder.add(Dense(inputs=200, activation='relu'))
encoder.add(Dense(inputs=100, activation='relu'))
encoder.add(Dense(inputs=50, activation='relu'))
encoder.add(Dense(inputs=25, activation='relu')) #The encoder

decoder = Sequential()
decoder.add(Dense(inputs=50, activation='relu', input_shape=[25]))
decoder.add(Dense(inputs=100, activation='relu'))
decoder.add(Dense(inputs=200, activation='relu'))
decoder.add(Dense(inputs=400, activation='relu'))
decoder.add(Dense(inputs=28*28, activation='relu'))
decoder.add(Reshape([28,28])

autoencoder = Sequential([encoder, decoder])
autoencoder.compile(loss='binary_crossentropy', optimizer=SGD(lr=1.5), metrics=['accuracy'])




