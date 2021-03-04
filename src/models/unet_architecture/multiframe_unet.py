import torch.nn as nn

from models.unet_architecture.unet_components import DoubleConvMF, DownMF, Up
from models.right_view.combine_fns import CombineConv3D, CombineMax, CombineAverage


class MultiFrameUNet(nn.Module):
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

        self.combine = self._determine_combine_fn(combine_fn)

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

    def _determine_combine_fn(self, combine_fn):
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
        pred_right = self.layers[-1](comb_xi[-1])

        return pred_right


# x = torch.rand(2, 5, 3, 100, 100)
# model = MultiFrameUNet(num_frames = 5, combine_fn="max")
# out = model(x)