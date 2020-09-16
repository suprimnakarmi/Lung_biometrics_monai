# Main file for lung biometrics using monai
import torch
import monai
import matplotlib.pyplot as plt
import os
import glob
import dataset
import data_transformation

# Path for all the images and labels in local device
image_path = "/home/suprim/dataset/MontgomerySet/CXR_png"
label_path = "/home/suprim/dataset/MontgomerySet/ManualMask/MergedMasks"

# Calling load_data_path function to load all the image files
images = dataset.load_data_path(image_path)
labels = dataset.load_data_path(label_path)

# Zipping the image and label to load it
data_dicts = [{"image": image, "label": label} for image,label in zip(images, labels)]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]
