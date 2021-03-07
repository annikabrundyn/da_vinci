import torch.nn as nn

# class CombineConv3D(nn.Module):
#     """
#     Combine multiple embeddings using a 3D CNN
#     """
#
#     def __init__(self, in_ch:int, out_ch: int = 1, kernel_size: int = 3, stride: int = 1, padding: int = 1):
#         super().__init__()
#         # in channels is num_frames
#         self.net = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
#
#     def forward(self, x):
#         x = self.net(x)
#
#         # [b, 1, d1, d2, d3] --> [b, d1, d2, d3]
#         x = x.squeeze(1)
#         return x
#
#
# class CombineConv1DAvgPool(nn.Module):
#     """
#     Combine multiple embeddings using a 3D CNN
#     """
#
#     def __init__(self, in_ch: int, out_ch: int = 1, temporal_kernel_size: int = 3, temporal_stride: int = 1, temporal_padding: int = 0):
#         super().__init__()
#         # in channels and out channels are the same as feature dim
#         self.net = nn.Conv1d(in_ch, out_ch,
#                              kernel_size=[temporal_kernel_size, 1, 1],
#                              stride=[temporal_stride, 1, 1],
#                              padding=[temporal_padding, 0, 0])
#
#     def forward(self, x):
#         x = self.net(x)
#
#         # [b, 1, d1, d2, d3] --> [b, d1, d2, d3]
#         x = x.squeeze(1)
#         return x


class CombineMax(nn.Module):
    # same as applying 3D Maxpooling over the temporal dimension
    def forward(self, x):
        x = x.max(dim=1)[0]
        return x


class CombineAverage(nn.Module):
    def forward(self, x):
        x = x.mean(dim=1)
        return x

# import torch
# x = torch.rand(16, 5, 64, 100, 100)
# model = CombineConv1D(64, 64)
#
# y = model(x)




