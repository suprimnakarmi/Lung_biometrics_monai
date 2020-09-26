# Defining models to train dataset( ResUnet,

from monai.networks.layers import Norm
from monai.networks.nets import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F

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
def single_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding= 1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace= True),
    )
    return conv

# Applying SegNet as per the research paper
class SegNet(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(SegNet, self).__init__()
        # Defining the stages for encoder of the network
        self.down_conv_11 = single_conv(input_channel,64)
        self.down_conv_12 = single_conv(64,64)

        self.down_conv_21 = single_conv(64, 128)
        self.down_conv_22 = single_conv(128, 128)

        self.down_conv_31 = single_conv(128, 256)
        self.down_conv_32 = single_conv(256, 256)
        self.down_conv_33 = single_conv(256, 256)

        self.down_conv_41 = single_conv(256, 512)
        self.down_conv_42 = single_conv(512, 512)
        self.down_conv_43 = single_conv(512, 512)

        self.down_conv_51 = single_conv(512, 512)
        self.down_conv_52 = single_conv(512, 512)
        self.down_conv_53 = single_conv(512, 512)

        # Defining the stages for decoder of the network
        self.down_conv_53d = single_conv(512, 512)
        self.down_conv_52d = single_conv(512, 512)
        self.down_conv_51d = single_conv(512, 512)

        self.down_conv_43d = single_conv(512, 512)
        self.down_conv_42d = single_conv(512, 512)
        self.down_conv_41d = single_conv(512, 256)

        self.down_conv_33d = single_conv(256, 256)
        self.down_conv_32d = single_conv(256, 256)
        self.down_conv_31d = single_conv(256, 128)

        self.down_conv_22d = single_conv(128, 128)
        self.down_conv_21d = single_conv(128, 64)

        self.down_conv_12d = single_conv(64, 64)
        self.down_conv_11d = nn.Conv2d(64, output_channel, kernel_size= 3, padding=1)
        self.Dropout = nn.Dropout(0.5)

    def forward(self,x):
# Stage 1
        x11 = self.down_conv11(x)
        x11 = self.Dropout(x11)
        x12 = self.down_conv12(x11)
        x1p, id1 = F.max_pool2d(x12, kernel_size = 2 , stride = 2, return_indices = True)

# Stage 2
        x21 = self.down_conv_21(x1p)
        x22 = self.down_conv_22(x21)
        x2p, id2 = F.max_pool2d(x22, kernel_size= 2 , stride =2, return_indices = True)

# Stage 3
        x31 = self.down_conv_31(x2p)
        x31 = self.Dropout(x31)
        x32 = self.down_conv_32(x31)
        x33 = self.down_conv_33(x32)
        x3p, id3 = F.max_pool2d(x33, kernel_size= 2 , stride =2, return_indices = True)

# Stage 4
        x41 = self.down_conv_41(x3p)
        x42 = self.down_conv_42(x41)
        x43 = self.down_conv_43(x42)
        x4p, id4 = F.max_pool2d(x43, kernel_size= 2 , stride =2, return_indices = True)

# Stage 5
        x51 = self.down_conv_51(x4p)
        x51 = self.Dropout(x51)
        x52 = self.down_conv_52(x51)
        x53 = self.down_conv_53(x52)
        x5p, id5 = F.max_pool2d(x53, kernel_size= 2 , stride =2, return_indices = True)

# Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size = 2, stride= 2)
        x53d = self.down_conv_53d(x5d)
        x52d = self.down_conv_52d(x53d)
        x51d = self.down_conv_51d(x52d)

# Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size = 2, stride= 2)
        x43d = self.down_conv_43d(x4d)
        x42d = self.down_conv_42d(x43d)
        x41d = self.down_conv_41d(x42d)

# Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size = 2, stride= 2)
        x33d = self.down_conv_33d(x3d)
        x32d = self.down_conv_32d(x33d)
        x31d = self.down_conv_31d(x32d)

# Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size = 2, stride= 2)
        x22d = self.down_conv_22d(x2d)
        x21d = self.down_conv_21d(x22d)

# Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size = 2, stride= 2)
        x12d = self.down_conv_12d(x1d)
        x11d = self.down_conv_11d(x12d)

        return x11d




