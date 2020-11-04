from unet_depth import UNetDepth

class UNetDirect(UNetDepth):
    def __init__(
        self,
        num_classes: int,
        input_channels: int,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
    ):
        super().__init__(num_classes,
        input_channels,
        num_layers,
        features_start,
        bilinear)

        self.conv_reshape = None

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        # Final conv layer of UNetDepth
        output = self.layers[-1](xi[-1])

        return output