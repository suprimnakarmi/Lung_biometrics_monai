# Defining models to train dataset( ResUnet,

from monai.networks.layers import Norm
from monai.networks.nets import UNet
import torch
import torch.nn as nn

# Set device to cpu if no Gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiating the object of UNet model
model_unet = UNet(
    dimensions= 2,
    in_channels=1,
    out_channels= 2,
    channels = (16,32,64,128,256),
    strides = (2,2,2,2),
    num_res_units=2,
    norm= Norm.BATCH,
).to(device)

# A function for double convolution in all the stages
def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding= 1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace= True),
        nn.Conv2d(out_c, out_c, kernel_size =3, padding = 1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace= True)
    )

class SegNet()

