"""
This module uses CNNs with Tensor & Keras to identify handwritten digits in the MNIST dataset.

In TensorFlow Keras, the `filters` parameter in the `Conv2D` layer represents the number of learnable convolutional
kernels or filters that will be applied to the input data. These filters are small grids that slide over the 
input data to detect patterns and features at different spatial locations.

Each filter is a set of learnable weights that are adjusted during the training process to capture specific 
features in the input data. The number of filters you specify determines how many different types of 
features or patterns the convolutional layer can detect in the input.

For example, if you have an input image with three color channels (e.g., red, green, and blue), 
and you use 32 filters in a `Conv2D` layer, it means that the layer will learn 32 different 
sets of weights to convolve with the input data. Each filter will be responsible for detecting 
a specific feature or pattern in the image. These patterns could be simple edges, textures, 
or more complex features as the network learns.

In practice, you often increase the number of filters in deeper layers of a convolutional neural network (CNN)
to allow the network to capture more abstract and complex features as you progress through the network's layers.
The choice of the number of filters is a hyperparameter that you can tune based on your specific problem and 
dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping


##########################################################
# Process the data - 28x28 pixel greyscale images of numbers
#########################################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#One-hot encode labels
y_cat_test = to_categorical(y_test, num_classes=10)
y_cat_train = to_categorical(y_train, num_classes=10)

#To avoid gradient problems, normalise training data
x_test = x_test/255
x_train = x_train/255

#print(x_train.shape)
#Add a colour channel as the images are greyscale
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

#############################################################
# Build the model
#############################################################
model = Sequential()
#padding='valid' == no padding as the stride is 1 and 28/4 = 7 a whole number, 4 from kernel size 
#input_shape=(28,28,1) == the size of one image
model.add(
    Conv2D(filters=32, kernel_size=(4,4), strides=(1,1), padding='valid', input_shape=(28,28,1), activation='relu')
    )
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten()) #28*28 -> 784 1D array
model.add(Dense(units=128, activation='relu'))
#Output layer  multi-class -> softmax
model.add(Dense(units=10, activation='softmax')) #units=10 as 10 classes, activation='softmax' as multiple classification

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=1) # patience=1 == wait one epoch

model.fit(x_train, y_cat_train, epochs=10, validation_data=(x_test, y_cat_test), callbacks=[early_stop])