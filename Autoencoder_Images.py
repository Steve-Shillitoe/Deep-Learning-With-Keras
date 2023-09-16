import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, GaussianNoise
from tensorflow.keras.optimizers import SGD

#####################################################
# Image processing
####################################################
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#scale images
X_train = X_train/255
X_test = X_test/255

#########################################################################
#Build stacked autoencoder that reduces the dimensionality of the dataset
#########################################################################
encoder = Sequential()
encoder.add(Flatten(input_shape=[28,28]))
encoder.add(Dense(units=400, activation='relu'))
encoder.add(Dense(units=200, activation='relu'))
encoder.add(Dense(units=100, activation='relu'))
encoder.add(Dense(units=50, activation='relu'))
encoder.add(Dense(units=25, activation='relu')) #The encoder

decoder = Sequential()
decoder.add(Dense(units=50, activation='relu', input_shape=[25]))
decoder.add(Dense(units=100, activation='relu'))
decoder.add(Dense(units=200, activation='relu'))
decoder.add(Dense(units=400, activation='relu'))
decoder.add(Dense(units=28*28, activation='relu'))
decoder.add(Reshape([28,28]))

autoencoder = Sequential([encoder, decoder])
autoencoder.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=1.5), metrics=['accuracy'])
autoencoder.fit(X_train, X_train, epochs=5, validation_data=[X_test, X_test])
#reconstructed images after reducing dimensionality down to 25 neurons
passed_images = autoencoder.predict(X_test[:10])  

n=0
plt.title("Original image")
plt.imshow(X_test[n])
plt.show()
plt.title("Reconstructed image")
plt.imshow(passed_images[n])
plt.show()

###################################################
## Add noise to images
##################################################





