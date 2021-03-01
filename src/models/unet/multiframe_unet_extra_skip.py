import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet.unet_components import DoubleConvMF, DownMF, Up
from models.right_view.multiframe.combine_fns import CombineConv3D, CombineMax, CombineAverage


class MultiFrameUNetExtraSkip(nn.Module):
    """
    Args:
        num_classes: Number of output classes required
        num_layers: Number of layers in each side of U-net (default 5)
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
        self.input_channels = input_channels
        self.num_frames = num_frames

        self.combine = self.determine_combine(combine_fn)

        layers = [DoubleConvMF(self.input_channels, features_start)]
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

        # WIP: should we combine them with a single convolutional layer? no non-linearity?
        self.final_conv = nn.Conv2d(2 * output_channels, output_channels, kernel_size=3, padding=1)

    def determine_combine(self, combine_fn):
        if combine_fn == "conv_3d":
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
        orig_output = self.layers[-1](comb_xi[-1])

        # WIP: add additional connection straight from input to output
        # TODO: double check that the first frame is the "latest frame" -- I THINK ITS THE LAST FRAME ACTUALLY
        input = x[:, 0, :, :, :]
        input_concat_output = torch.cat([input, orig_output], dim=1)
        new_output = self.final_conv(input_concat_output)

        # TODO: output both orig_output and new_output to visualize the shift

        return new_output


x = torch.rand(2, 5, 3, 100, 100)
model = MultiFrameUNetExtraSkip(num_frames = 5, combine_fn="max")
out = model(x)