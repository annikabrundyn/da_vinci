import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet.unet_components import DoubleConvMF, DownMF, Up


class CombineConv3D(nn.Module):
    """
    Combine multiple embeddings using a 3D CNN
    """

    def __init__(self, in_ch:int, out_ch: int = 1, kernel_size:int = 1, stride:int = 1):
        super().__init__()
        # in channels is num_frames
        self.net = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.net(x)

        # [b, 1, d1, d2, d3] --> [b, d1, d2, d3]
        x = x.squeeze(1)
        return x


class CombineMax(nn.Module):
    # same as applying 3D Maxpooling over the temporal dimension
    def forward(self, x):
        x = x.max(dim=1)[0]
        return x


class CombineAverage(nn.Module):
    def forward(self, x):
        x = x.mean(dim=1)
        return x







