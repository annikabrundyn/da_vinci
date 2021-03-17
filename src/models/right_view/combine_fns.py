import torch.nn as nn
from convlstm import ConvLSTM


class CombineConvLSTM(nn.Module):
    """
    Combine multiple embeddings using a ConvLSTM layer
    """

    def __init__(self,
                 channels: int,
                 kernel_size: tuple = (3, 3),
                 padding: int = 1,
                 num_layers: int = 1,
                 batch_first: bool = True,
                 bias: bool = True
                 ):
        super().__init__()
        # in channels and out channels stay the same
        self.net = ConvLSTM(channels, channels, kernel_size=kernel_size, padding=padding,
                            num_layers=num_layers, batch_first=batch_first, bias=bias)

    def forward(self, x):
        # [b, t, c, h, w] --> [b, c, t, h, w]
        x = x.permute(0, 2, 1, 3, 4)

        # [b, c, t, h, w] --> [b, c, d1, d2]
        output = self.net(x)

        return x[:, -1, ...]


class CombineConv3D(nn.Module):
    """
    Combine multiple embeddings using a 3D CNN
    input shape:
    """

    def __init__(self,
                 channels: int,
                 num_frames: int,
                 spatial_kernel_size: int = 3,
                 spatial_stride: int = 1,
                 spatial_padding: int = 1
                 ):
        super().__init__()
        # in channels and out channels stay consist
        kernel_size = [num_frames, spatial_kernel_size, spatial_kernel_size]
        stride = [1, spatial_stride, spatial_stride]
        padding = [0, spatial_padding, spatial_padding]

        self.net = nn.Conv3d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # [b, t, c, h, w] --> [b, c, t, h, w]
        x = x.permute(0, 2, 1, 3, 4)

        x = self.net(x)

        # [b, c, t, h, w] --> [b, t, c, h, w]
        x = x.permute(0, 2, 1, 3, 4)

        # [b, 1, d1, d2, d3] --> [b, d1, d2, d3]
        x = x.squeeze(1)
        return x


class CombineMax(nn.Module):
    # same as applying 3D Maxpooling over the temporal dimension
    def __init__(self, channels, num_frames):
        super().__init__()

    def forward(self, x):
        x = x.max(dim=1)[0]
        return x


class CombineAverage(nn.Module):
    def __init__(self, channels, num_frames):
        super().__init__()

    def forward(self, x):
        x = x.mean(dim=1)
        return x





