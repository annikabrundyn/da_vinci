import os.path
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import ssim, psnr
import lpips

from losses import Perceptual, L1_Perceptual, L1_SSIM
from metrics import FIDCallback

from data.multiframe_data import MFDaVinciDataModule
from models.right_view.multiframe import MultiFrameUNet


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        num_frames: int,
        combine_fn: str,
        loss: str,
        num_layers: int,
        bilinear: bool,
        features_start: int = 64,
        lr: float = 0.001,
        log_tb_imgs: bool = True,
        tb_img_freq: int = 10000,
        **kwargs
    ):
        super().__init__()
        # self.save_hyperparameters()
        # self.num_frames = num_frames
        # self.combine_fn = combine_fn
        # self.loss = loss
        #
        # self.criterion = self._determine_loss_fn()
        #
        # # by default assuming color input and output (3 channels)
        # self.net = MultiFrameUNet(num_frames=num_frames,
        #                           combine_fn=combine_fn,
        #                           num_layers=num_layers,
        #                           features_start=features_start,
        #                           bilinear=bilinear)
        #
        # self.LPIPS = lpips.LPIPS(net='alex')

    def _determine_loss_fn(self):
        if self.loss == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = torch.nn.MSELoss()
        elif self.loss == "ssim":
            self.criterion = SSIM()
        elif self.loss == "perceptual":
            self.criterion = Perceptual()
        elif self.loss == "l1_perceptual":
            self.criterion = L1_Perceptual()
        elif self.loss == "l1_ssim":
            self.criterion = L1_SSIM()
        else:
            print("Using MSE Loss")
            self.criterion = torch.nn.MSELoss()

        return self.criterion

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # predict right view
        img, target = batch
        pred = self(img)

        # calculate loss
        loss_val = self.criterion(pred, target)

        # calculate metrics
        ssim_val = ssim(pred, target)
        psnr_val = psnr(pred, target)
        lpips_val = self.LPIPS(pred, target).mean()

        # log loss + metrics
        logs = {'train_loss': loss_val, 'train_ssim': ssim_val, 'train_psnr': psnr_val, 'train_lpips': lpips_val}
        self.log_dict(logs)

        # log predicted images
        if self.hparams.log_tb_imgs and self.global_step % self.hparams.tb_img_freq == 0:
            # pick random element in batch to visualize
            idx = np.random.choice(len(img))
            self._log_images(img[idx], target[idx], pred[idx], step_name="train")

        return loss_val

    def validation_step(self, batch, batch_idx):
        # predict right view
        img, target = batch
        pred = self(img)

        # calculate loss
        loss_val = self.criterion(pred, target)

        # calculate metrics
        ssim_val = ssim(pred, target)
        psnr_val = psnr(pred, target)
        lpips_val = self.LPIPS(pred, target).mean()

        # log loss + metrics
        logs = {'val_loss': loss_val, 'val_ssim': ssim_val, 'val_psnr': psnr_val, 'val_lpips': lpips_val}
        self.log_dict(logs)

        # log predicted images
        if self.hparams.log_tb_imgs and self.global_step % self.hparams.tb_img_freq == 0:
            # pick random element in batch to visualize
            idx = np.random.choice(len(img))
            self._log_images(img[idx], target[idx], pred[idx], step_name="val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [opt]

    def _log_images(self, img, target, pred, step_name):

        self.logger.experiment.add_image(f'{step_name}_input_left', make_grid(img), self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_target_right', make_grid(target), self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_pred_right', make_grid(pred), self.trainer.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Required arguments
        parser.add_argument("--data_dir", type=str, help="path to davinci data")
        parser.add_argument("--num_frames", type=int, help="number of consecutive frames to include")
        parser.add_argument("--combine_fn", type=str, help="how to combine multiple frames")
        parser.add_argument("--loss", type=str, choices=['l1', 'mse', 'ssim', 'perceptual', 'l1_perceptual', 'L1_SSIM'], help="loss function")
        parser.add_argument("--bilinear", action='store_true', help="bilinear (True) vs. transposed convolution (False)")
        parser.add_argument("--num_layers", type=int, help="number of layers/blocks in u-net")

        # hyperparameters with a default value
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument("--num_workers", type=int, default=8)

        # logging
        parser.add_argument("--log_tb_imgs", action='store_true', default=True)
        parser.add_argument("--tb_img_freq", type=int, default=10000)
        parser.add_argument("--save_img_freq", type=int, default=50)
        parser.add_argument("--fid_freq", type=int, default=500)
        return parser

