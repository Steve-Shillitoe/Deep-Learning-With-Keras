"""
In this module  TensorFlow 2 and Keras are used to build a 
Convolutional Neural Network that 
can identify cells with the maleria parasite.  

Real images of varying size are used to train the model.
"""

from random import shuffle
from numpy._typing import _128Bit
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

def random_brightness_and_contrast(image):
    # This function applies random brightness and contrast adjustments to images

    # Randomly adjust brightness
    image = tf.image.random_brightness(image, max_delta=0.2)  # You can adjust the max_delta value

    # Randomly adjust contrast
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # You can adjust the lower and upper values

    return image

############################################################
### Reading in the data
############################################################
test_path = 'cell_images\\test\\'
train_path = 'cell_images\\train\\'

#images may have varying sizes, so find the average dimensions
dim1 = []
dim2 = []

for image_filename in os.listdir(test_path + 'uninfected'):
    img = imread(test_path + 'uninfected\\' + image_filename)
    d1, d2, colours = img.shape
    dim1.append(d1)
    dim2.append(d2)
    
#print('dim1 mean = {} dim2 mean = {}'.format(np.mean(dim1), np.mean(dim2)))
#output of above = dim1 mean = 130.92538461538462 dim2 mean = 130.75
#so a mean diminsion of 130 is sensible
image_shape = (130, 130, 3)

#############################################################
### Image Processing with ImageDataGenerator
#############################################################
# The primary purpose of ImageDataGenerator is to perform data augmentation on images. 
# Data augmentation is a technique used to artificially increase the size of your training 
# dataset by applying various transformations to the existing images. 
# These transformations can include rotations, flips, shifts, zooms, and more.
# A preprocessing function is also used to randomly adjust image brightness & contrast
# Data augmentation helps improve the model's generalization by 
# exposing it to a wider range of variations in the data.
image_gen = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode='nearest',
                               preprocessing_function=random_brightness_and_contrast)

#image_gen.flow_from_directory(train_path)

###################################################################################
### Create the model
###################################################################################
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))

model.add(Dropout(0.5)) #Turn off half the neurons to prevent overfitting

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2)

batch_size = 16

train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=image_shape[:2],
                                                color_mode = 'rgb',
                                                batch_size = batch_size,
                                                class_mode = 'binary')  

test_image_gen = image_gen.flow_from_directory(test_path,
                                                target_size=image_shape[:2],
                                                color_mode = 'rgb',
                                                batch_size = batch_size,
                                                class_mode = 'binary',
                                                shuffle=False)  

# model.fit(train_image_gen,
#         epochs=20, 
#         validation_data=test_image_gen,
#         callbacks=[early_stop])

#To save time, load an existing model
model = load_model('malaria_detector.h5')

###################################################################################
### Evaluate the model
###################################################################################
prediction = model.predict(test_image_gen)
prediction = prediction > 0.5
print('prediction = {}'.format(prediction))
print('\n classification report = ', classification_report(test_image_gen.classes, prediction))
print('\n confusion matrix = ', confusion_matrix(test_image_gen.classes, prediction))

test_img = image.load_img('cell_images\\train\parasitized\\C100P61ThinF_IMG_20150918_144104_cell_162.png',
               target_size=image_shape)
test_img_arr = image.img_to_array(test_img)
test_img_arr = np.expand_dims(test_img_arr, axis=0)
print('Test image shape = ',test_img_arr.shape)
print('Try to predict a parasitized cell - ', model.predict(test_img_arr))
print('\n')
print('class indices = ', train_image_gen.class_indices)




