#Main file for lung biometrics using monai
import torch
import monai
import matplotlib.pyplot as plt
import os
import glob
import dataset

image_path = "/home/suprim/dataset/MontgomerySet/CXR_png"
label_path = "/home/suprim/dataset/MontgomerySet/ManualMask/MergedMasks"

images = dataset.load_data_path(image_path)
labels = dataset.load_data_path(label_path)
