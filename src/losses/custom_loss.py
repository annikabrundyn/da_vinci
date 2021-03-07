from pytorch_lightning.metrics import SSIM
from losses.loss_registry import LossRegistry
from losses.perceptual_loss import Perceptual
from torch import nn
import torch.nn.functional as F

from losses.perceptual_loss import Perceptual
from pytorch_lightning.metrics.functional import ssim
import torch
import abc


class CustomLoss(nn.Module):
    def reset(self):
        pass


# @LossRegistry.register('l1_perceptual')
class L1_Perceptual(CustomLoss):
    def __init__(self) -> None:
        '''
        Equal weight for L1 + Perceptual Loss
        '''
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = Perceptual()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self.l1(y_true, y_pred) + self.perceptual(y_true, y_pred)


@LossRegistry.register('l1_ssim')
class L1_SSIM(CustomLoss):
    def __init__(self) -> None:
        '''
        Equal weight for L1 + SSIM
        '''
        super().__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(y_true, y_pred) + ssim(y_true, y_pred)

