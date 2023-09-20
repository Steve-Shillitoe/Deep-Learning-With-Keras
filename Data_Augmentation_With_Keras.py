"""
This module demonstrates the use of the Keras ImageDataGenerator to generate 
augmented images from a smaller set of original images.


Image shifts via the width_shift_range and height_shift_range arguments.
Image flips via the horizontal_flip and vertical_flip arguments.
Image rotations via the rotation_range argument
Image brightness via the brightness_range argument.
Image zoom via the zoom_range argument.
"""
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage import io
import os
from PIL import Image

def random_brightness_and_contrast(image):
    # This function applies random brightness and contrast adjustments to images

    # Randomly adjust brightness
    image = tf.image.random_brightness(image, max_delta=0.9)  # You can adjust the max_delta value

    # Randomly adjust contrast
    image = tf.image.random_contrast(image, lower=0.2, upper=2.5)  # You can adjust the lower and upper values

    return image

# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor. 

datagen = ImageDataGenerator(
        brightness_range=[0.2, 2.5],  # Adjust brightness randomly between 0.7 (darker) and 1.3 (brighter)
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect',  #Also try nearest, constant, reflect, wrap 
        cval=125)
        #preprocessing_function=random_brightness_and_contrast)   

# ####################################################################
# #Multiple images.
# #Manually read each image and create an array to be supplied to datagen
# via flow method

image_directory = 'Ferret_Images/Originals/'
SIZE = 128
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    image = io.imread(image_directory + image_name)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((SIZE,SIZE))
    dataset.append(np.array(image))

x = np.array(dataset)
#Let us save images to get a feel for the augmented images.
#Create an iterator either by using image dataset in memory (using flow() function)
#or by using image dataset from a directory (using flow_from_directory)
#from directory can be useful if subdirectories are organized by class
   
#Again, flow generates batches of randomly augmented images
  
i = 0
for batch in datagen.flow(x, batch_size=6,  
                          save_to_dir='Ferret_Images/Augmented/', 
                          save_prefix='aug', 
                          save_format='jpg'):
    i += 1
    if i > 20: # with batch_size=6, 126 images are generated from 6 images
        break  # otherwise the generator would loop indefinitely  



#This approach will not work with my folder structure because the Originals folder
#does not have sub-folders such as Ferrets & Cats
# i = 0
# for batch in datagen.flow_from_directory(directory='Ferret_Images\\Originals\\', 
#                                          batch_size=16,  
#                                          target_size=(256, 256),
#                                          color_mode="rgb",
#                                          save_to_dir='Ferret_Images\\Augmented\\', 
#                                          save_prefix='aug', 
#                                          save_format='jpg',
#                                          class_mode=None):
#     i += 1
#     if i > 31:
#         break 

#Creates 32 images for each class. 




######################################################################
#Loading a single image for demonstration purposes.
#Using flow method to augment the image

# Loading a sample image  
#Can use any library to read images but they need to be in an array form
#If using keras load_img convert it to an array first
#x = io.imread("Ferret_Images/Originals/Syrup.jpg")  #Array with shape (793, 551, 3)

# Reshape the input image because ...
#x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
#First element represents the number of images
#x = x.reshape((1, ) + x.shape)  #Array with shape (1, 793, 551, 3)

# i = 0
# for batch in datagen.flow(x, batch_size=16,  
#                           save_to_dir='Ferret_Images/Augmented', 
#                           save_prefix='aug', 
#                           save_format='jpg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely  
