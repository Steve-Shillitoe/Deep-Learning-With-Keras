"""
This module demonstrates the use of the Augmentor package to generate a large set of
augmented images from a small sample of images. 

It is possible to define the number of images that will be generated and they are 
created in a folder called output within the folder of orginal images.
"""

import Augmentor

#Specify the folder containing the original images
p = Augmentor.Pipeline("Ferret_Images/Originals")  #contains 6 ferret images.

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
p.shear(probability=0.5, max_shear_left=2, max_shear_right=2)
p.random_brightness(probability=0.3, min_factor=0.3, max_factor=1.2)
p.random_contrast(probability=0.3, min_factor=0.3, max_factor=1.2)
p.random_color(probability=0.2, min_factor=0.0, max_factor=0.75) # 0=b+W, 1=original image

p.sample(60)  #generate 60 augmented images from the original images.
