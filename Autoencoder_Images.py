"""
In this module an autoencoder is created that removes noise from an image.

Noise is added to the images of handwritten digits 0-9 in the MNIST dataset
using the GaussianNoise class.  The autoencoder is then used to remove this noise.
"""
#import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, GaussianNoise
#from tensorflow.keras.optimizers import SGD

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
tf.random.set_seed(101)
encoder = Sequential()
encoder.add(Flatten(input_shape=[28,28]))
encoder.add(GaussianNoise(0.2))  #Add noise to images
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

noise_remover = Sequential([encoder, decoder])
noise_remover.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
noise_remover.fit(X_train, X_train, epochs=10, validation_data=[X_test, X_test])
#reconstructed images after reducing dimensionality down to 25 neurons
passed_images = noise_remover.predict(X_test[:10])  

sample = GaussianNoise(0.2)
ten_noisy_images = sample(X_test[:10], training=True)
denoised = noise_remover(ten_noisy_images)

n=0
plt.title("Original image")
plt.imshow(X_test[n])
plt.show()
plt.title("Noisy image")
plt.imshow(ten_noisy_images[n])
plt.show()
plt.title("Cleaned image")
plt.imshow(denoised[n])
plt.show()





