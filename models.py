# Defining models to train dataset( ResUnet,

from monai.networks.layers import Norm
from monai.networks.nets import UNet
import torch

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

