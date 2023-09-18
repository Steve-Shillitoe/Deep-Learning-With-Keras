"""
In this module a CNN is used to make an image classifyer that 
can classify the 10 types of clothing in the MNIST Fashion dataset of images.
"""
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd

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

model.add(Conv2D( filters=32, kernel_size=(4,4), strides=(1,1), padding='valid', input_shape=(28,28,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten()) #28*28 -> 784 1D array
model.add(Dense(units=128, activation='relu'))
#Output layer  multi-class -> softmax
model.add(Dense(units=10, activation='softmax')) #units=10 as 10 classes, activation='softmax' as multiple classification

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#################################################################
# Train the model
#################################################################
model.fit(x_train, y_cat_train, epochs=5, validation_data=(x_test, y_cat_test))

#################################################################
# Evaluate the model
#################################################################
# In this section the accuracy, precision, recall, f1-score the model achieved 
# on the x_test data set will be evaluated

metrics = pd.DataFrame(model.history.history)
print('metrics = ', metrics)

metrics[['loss', 'val_loss']].plot()
metrics[['accuracy', 'val_accuracy']].plot()
plt.show()

y_pred = model.predict(x_test)
predicted_labels = np.argmax(y_pred, axis=1)
print('\n predicted_labels = {}'.format(predicted_labels))
print('\n classification_report = ', classification_report(y_test, predicted_labels))
