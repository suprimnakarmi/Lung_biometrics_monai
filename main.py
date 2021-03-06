# Main file for lung biometrics using monai
import torch
import monai
import matplotlib.pyplot as plt
import os
import glob
from monai.data import CacheDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss

import dataset
import data_transformation
from models import model_unet
from training import train

# Path for all the images and labels in local device
image_path = "/home/suprim/dataset/MontgomerySet/CXR_png"
label_path = "/home/suprim/dataset/MontgomerySet/ManualMask/MergedMasks"

# Calling load_data_path function to load all the image files
images = dataset.load_data_path(image_path)
labels = dataset.load_data_path(label_path)

# Zipping the image and label to load it
data_dicts = [{"image": image, "label": label} for image,label in zip(images, labels)]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]

# Transforming the dataset
train_ds = CacheDataset(data=train_files,
                        transform=data_transformation.train_transforms,
                        cache_rate=0.3,
                        num_workers=4)
val_ds = CacheDataset(data=val_files,
                      transform=data_transformation.val_transforms,
                      cache_rate=0.3,
                      num_workers=4)

# Loading the dataset
train_loader = DataLoader(train_ds, batch_size= 5, shuffle= True, num_workers=4)
val_loader= DataLoader(val_ds, batch_size = 1)

loss_function = DiceLoss(to_onehot_y= True, softmax= True)
optimizer = torch.optim.Adam(model_unet.parameters(), lr=1e-4)
epoch_num=10
val_interval = 2

writer = SummaryWriter("/home/suprim/dataset/MontgomerySet/Run")

train(epoch_num, val_interval, train_loader, val_loader, writer, train_ds, loss_function, optimizer)
