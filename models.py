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
        self.down_conv_1 = single_conv(64, output_channel)

    def forward(self,x):
# Stage 1
        x11 = self.down_conv11(x)
        x12 = self.down_conv12(x11)
        x1p, id1 = F.max_pool2d(x12, kernel_size = 2 , stride = 2, return_indices = True)

# Stage 2
        x21 = self.down_conv_21()





