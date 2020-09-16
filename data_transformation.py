# Define data transformation for training and validation dataset
from monai.transforms import(
    AddChanneld,
    Compose,
    LoadPNGd,
    ScaleIntensityRanged,
    RandAffined,
    ToTensord,
    CropForegroundd,
    RandFlipd,
    RandGaussianNoised,
    Resized
)
import numpy as np


train_transforms = Compose(
    [
        LoadPNGd(keys= ["image", "label"]),
        AddChanneld(keys= ["image", "label"]),
        ScaleIntensityRanged(keys = ["image", "label"], a_min=0, a_max= 255, b_min= 0.0, b_max= 1.0, clip = True),
        CropForegroundd(keys = ["image", "label"], source_key="image"),
        RandGaussianNoised(keys= ["image", "label"], prob=0.5, mean=0.0,std=0.1),
        RandFlipd(keys=["image", "label"], spatial_axis = 0, prob=0.2),
        RandAffined(
            keys =["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.7,
            spatial_size=(1024, 1024),
            translate_range=(60,40),
            rotate_range=(np.pi/ 30, np.pi / 30),
            scale_range=(0.15, 0.15),
            padding_mode="border",
        ),
        ToTensord(keys=["image", "label"]

        )
    ]
)

val_transforms = Compose(
    [
        LoadPNGd(keys=["image", "label"]),
        AddChanneld(keys= ["image", "label"]),
        ScaleIntensityRanged(keys=["image", "label"], a_min=0, a_max=255, b_min= 0.0, b_max=1.0,clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image","label"], spatial_size=(1024, 1024)),
        ToTensord(keys=["image", "label"])
    ]
)



