# -*- coding: utf-8 -*-

from hm_fileconvert import fileconvert
from hm_spectogram import preprocessing

# Example usage:
# Define the directory path and cropping parameters
dir_path = r"/Users/songsooyeon/Desktop/SD/매미탐사_8월1일_15일"
crop_height_low = 3000
crop_height_high = 20000
crop_sec = 5

# Create an instance of the preprocessing class
processor = preprocessing(dir_path, crop_height_low, crop_height_high, crop_sec)

# Process audio and generate spectrograms
# processor.process_audio()

image_constant, image_list = processor.load_image()
width, height, height_high, height_low, width_crop = processor.define_constants(image_constant)
processor.crop_images(image_list, width, height, height_high, height_low, width_crop)

# save_path = r"/Users/songsooyeon/Desktop/SD/매미탐사_8월1일_15일/input"
# photo_path = r"/Users/songsooyeon/Desktop/SD/매미탐사_8월1일_15일/spectogram"

# processor = preprocessing(save_path, photo_path)
processor.process_images()

# /Users/songsooyeon/Desktop/SD/매미탐사_8월1일_15일/in/resized