import torch
import torch.nn as nn
import torch.nn.functional as F

### 2D UNET COMPONENTS ####


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """
    def __init__(self, in_ch: int, out_ch: int, batch_norm: bool = False):
        super().__init__()
        conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        relu = nn.ReLU(inplace=True)

        if batch_norm:
            bnorm = nn.BatchNorm2d(out_ch)
            convs = [conv1, bnorm, relu, conv2, bnorm, relu]
        else:
            convs = [conv1, relu, conv2, relu]

        self.net = nn.Sequential(*convs)

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, in_ch: int, out_ch: int, batch_norm: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch, batch_norm=batch_norm)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path, followed by DoubleConv.
    """
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


### 3D UNET COMPONENTS ###

class DoubleConvMF(DoubleConv):
    """
    Multiframe version
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, x):
        """
        x -> (batch, frames, channels, height, width)
        """
        # (b, f, c, h, w) -> (bf, c, h, w)
        b, f, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        x = self.net(x)

        # (bf, c, h, w) -> (b, f, c, h, w)
        _, c, h, w = x.size()
        x = x.view(b, f, c, h, w)
        return x


class DownMF(Down):
    """
    Multiframe version
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # (b, f, c, h, w) -> (bf, c, h, w)
        b, f, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        x = self.net(x)

        # (bf, c, h, w) -> (b, f, c, h, w)
        _, c, h, w = x.size()
        x = x.view(b, f, c, h, w)

        return x
