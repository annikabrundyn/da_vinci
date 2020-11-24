import torch
from models.depth_map.unet import UNet


class RightUNet(UNet):
    """
    Args:
        num_classes: Number of output classes required
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear (bool): Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """

    def __init__(
            self,
            num_classes: int = 3,
            input_channels: int = 3,
            num_stack_horizontal: int = 1,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False
    ):
        super().__init__(num_classes,
                         input_channels,
                         num_stack_horizontal,
                         num_layers,
                         features_start,
                         bilinear)
        self.conv_reshape = None
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        # Final conv layer of UNet
        output = self.layers[-1](xi[-1])
        output = self.sig(output)
        return output
