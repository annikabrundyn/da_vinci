import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, "enc_{:d}".format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, "enc_{:d}".format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class Perceptual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = VGG16FeatureExtractor()
        self.l1 = nn.L1Loss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        feat_pred = self.extractor(y_pred)
        feat_gt = self.extractor(y_true)

        return sum([self.l1(feat_pred[i], feat_gt[i]) for i in range(3)])
