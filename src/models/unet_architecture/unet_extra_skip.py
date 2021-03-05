import torch
import torch.nn as nn

from models.unet_architecture.unet_components import DoubleConv, Up, Down


class UNetExtraSkip(nn.Module):
    """
    Args:
        num_classes: Number of output classes required
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear (bool): Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """
    def __init__(
            self,
            input_channels: int,
            output_channels: int = 3,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_channels = input_channels

        layers = [DoubleConv(self.input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, output_channels, kernel_size=1))

        self.layers = nn.ModuleList(layers)

        # WIP: should we combine them with a single convolutional layer? no non-linearity?
        self.final_conv = nn.Conv2d(2 * output_channels, output_channels, kernel_size=3, padding=1)


    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        # Final conv layer of original UNet
        orig_output = self.layers[-1](xi[-1])

        # WIP: add additional connection straight from input to output
        # TODO: double check that the first frame is the "latest frame"
        input = x[:, 0:3, :, :]
        input_concat_output = torch.cat([input, orig_output], dim=1)
        new_output = self.final_conv(input_concat_output)

        # TODO: output both orig_output and new_output to visualize the shift

        return new_output
