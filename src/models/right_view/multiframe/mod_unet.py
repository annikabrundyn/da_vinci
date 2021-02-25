import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet.unet_components import DoubleConvMF, DownMF, Up


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

        layers = [DoubleConvMF(self.input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(DownMF(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def combine(self, x):
        '''
        :param x: [b, f, feat_dim]
        :return: [b, feat_dim]
        '''
        return x.mean(dim=1)

    def forward(self, x):

        ### Encoder
        xi = [self.layers[0](x)]
        comb_xi = [self.combine(xi[0])]

        for layer in self.layers[1:self.num_layers]:
            out_feats = layer(xi[-1])
            xi.append(out_feats)
            comb_xi.append(self.combine(out_feats))

        ### Decoder
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            #xi[-1] = layer(xi[-1], xi[-2 - i])
            comb_xi[-1] = layer(comb_xi[-1], comb_xi[-2 - i])

        # Final conv layer of UNet
        pred_right = self.layers[-1](comb_xi[-1])

        return pred_right


x = torch.rand(2, 5, 3, 100, 100)
model = MultiFrameUNet(num_classes=3, input_channels=3)
out = model(x)