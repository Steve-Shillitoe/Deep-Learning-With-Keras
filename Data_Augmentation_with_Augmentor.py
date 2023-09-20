import Augmentor

p = Augmentor.Pipeline("Ferret_Images/Originals")

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
p.random_brightness(probability=0.3, min_factor=0.3, max_factor=1.2)
p.random_contrast(probability=0.3, min_factor=0.3, max_factor=1.2)
p.random_color(probability=0.2, min_factor=0.0, max_factor=0.75) # 0=b+W, 1=original image

p.sample(50)
