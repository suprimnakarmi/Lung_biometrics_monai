#file to import dataset from the device
import os
import glob

#Function to load each file path of images
def load_data_path(file_path):
    files= sorted(glob.glob(os.path.join(file_path, "*.png")))
    return files