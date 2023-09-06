"""
The CIFAR-10 dataset consists of 60,000 small, color images, which are divided into 10 different classes.

Here are the 10 classes in the CIFAR-10 dataset:

Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck

Each image in the CIFAR-10 dataset is 32x32 pixels in size and is in color, 
meaning it has three color channels (red, green, and blue). 

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.datasets import cifar10

##########################################################
# Process the data 
#########################################################
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train/255   #Normalisation applied to all 3 colour channels
x_test = x_test/255

y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)

#######################################################
# Build the model
#######################################################
model = Sequential()

#First convolutional layer
model.add(
    Conv2D(filters=32, kernel_size=(4,4), strides=(1,1), padding='valid', input_shape=(32,32,3), activation='relu')
    )
model.add(MaxPool2D(pool_size=(2,2)))

#Second Convolutional layer as images somewhat complext
model.add(
    Conv2D(filters=32, kernel_size=(4,4), strides=(1,1), padding='valid', input_shape=(32,32,3), activation='relu')
    )
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Flatten()) #32*32 -> 1024 1D array

model.add(Dense(units=256, activation='relu'))

#Output layer  multi-class -> softmax
model.add(Dense(units=10, activation='softmax')) #units=10 as 10 classes, activation='softmax' as multiple classification

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

early_stop = EarlyStopping(monitor='val_loss', patience=2) # patience=1 == wait one epoch

model.fit(x_train, y_cat_train, epochs=15, validation_data=(x_test, y_cat_test), callbacks=[early_stop])


