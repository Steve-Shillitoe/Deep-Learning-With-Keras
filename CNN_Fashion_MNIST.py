from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

###############################################################
### Load the data
###############################################################
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#####################################################################
# Visualise the Data
#####################################################################
single_image = x_train[0]
plt.imshow(single_image) #, cmap=plt.cm.binary
plt.show()

####################################################################
# Preprocess the data
####################################################################
#normalise the data
x_train = x_train/x_train.max()
x_test = x_test/x_test.max()
print(x_train.shape, x_test.shape)

#reshape image data to include a colour channel. 
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

#Convert the y_train & y_test data to one hot encoded for categorical analysis by Keras
y_cat_test = to_categorical(y_test, 10)  #10 classes 0-9
y_cat_train = to_categorical(y_train, 10)  #10 classes 0-9

##################################################################
#Build the model
##################################################################
model = Sequential()