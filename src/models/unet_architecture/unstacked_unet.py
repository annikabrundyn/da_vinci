import torch
import torch.nn as nn

from models.unet_architecture.unet_components import DoubleConvMF, DownMF, Up
#from models.right_view.combine_fns import CombineConv3D, CombineMax, CombineAverage
from models.right_view.combine_fns import CombineMax, CombineAverage


class UnstackedUNet(nn.Module):
    """
    Args:
        num_frames: Number of consecutive video frames to use as input
        combine_fn: How to combine features across temporal dimension after each convolutional 'Down' block
        input_channels: Number of channels for input frames (3/color)
        output_channels: Number of channels in output (3/color)
        num_layers: Number of convolutional blocks in each side of UNet
        features_start: Number of features in first layer (default 64)
        bilinear (bool): Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """
    def __init__(
            self,
            num_frames: int,
            combine_fn: str,
            input_channels: int = 3,
            output_channels: int = 3,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_frames = num_frames

        self.combine = self._determine_combine_fn(combine_fn)

        layers = [DoubleConvMF(input_channels, features_start)]
        combine_modules = [self.combine]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(DownMF(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            combine_modules.append(self.combine)
            feats //= 2

        layers.append(nn.Conv2d(feats, output_channels, kernel_size=1))

        self.layers = nn.ModuleList(layers)
        self.combine_modules = nn.ModuleList(combine_modules)

    def _determine_combine_fn(self, combine_fn):
        if combine_fn == "conv3d":
            combine = CombineConv3D(self.num_frames)

        elif combine_fn == "max":
            combine = CombineMax()

        elif combine_fn == "average":
            combine = CombineAverage()

        return combine

    def forward(self, x):

        ### Encoder
        xi = [self.layers[0](x)]
        comb_xi = [self.combine_modules[0](xi[0])]

        for enc_layer, combine in zip(self.layers[1:self.num_layers], self.combine_modules[1:]):
            out_feats = enc_layer(xi[-1])
            xi.append(out_feats)
            comb_xi.append(combine(out_feats))

        ### Decoder
        for i, dec_layer in enumerate(self.layers[self.num_layers:-1]):
            comb_xi[-1] = dec_layer(comb_xi[-1], comb_xi[-2 - i])

        # Final conv layer of UNet
        pred_right = self.layers[-1](comb_xi[-1])

        return pred_right
