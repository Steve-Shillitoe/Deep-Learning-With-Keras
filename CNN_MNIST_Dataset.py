import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


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
x_train = x_train.reshape(6000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

