import torch
import torch.nn as nn
import torch.nn.functional as F


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
            num_classes: int,
            input_channels: int,
            num_stack_horizontal: int = 1,
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

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))


        self.layers = nn.ModuleList(layers)

        # final layer to match the depth map dimensions
        self.conv_reshape = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,11), stride=(1, num_stack_horizontal), padding=(1, num_stack_horizontal), dilation=(8, 8*num_stack_horizontal))

    def forward(self, x):

        # reshape x: [b, f, c, h, w] --> [b*f, c, h, w]
        b, f, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        ### Encoder
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))

        ### Decoder - Up path
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        # Final conv layer of UNet
        orig_output = self.layers[-1](xi[-1])
        # Extra conv layer to reshape for depth map dimensions
        reshaped = self.conv_reshape(orig_output)
        return reshaped, orig_output


