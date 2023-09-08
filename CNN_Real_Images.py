from numpy._typing import _128Bit
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
# Data augmentation helps improve the model's generalization by exposing it to a wider range of variations in the data.
image_gen = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode='')

#image_gen.flow_from_directory(train_path)

###################################################################################
### Create the model
###################################################################################


    


