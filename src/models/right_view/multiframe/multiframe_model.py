import torch
import torch.nn.functional as F
import torchvision

import os.path
import numpy as np

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import ssim, psnr

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from data.right_data import RightDaVinciDataModule
from models.right_view.multiframe.multiframe_unet import MultiFrameUNet
from losses import Perceptual, L1_Perceptual


class MultiFrameModel(pl.LightningModule):
    def __init__(
        self,
        num_frames: int,
        combine_fn: str,
        loss: str,
        num_layers: int,
        bilinear: bool,
        features_start: int = 64,
        lr: float = 0.001,
        log_tb_imgs: bool = False,
        tb_img_freq: int = 10000,
        **kwargs
    ):
        super().__init__()
        self.num_frames = num_frames
        self.combine_fn = combine_fn
        self.loss = loss

        self.save_hyperparameters()

        # by default assuming color input and output (3 channels)
        self.net = MultiFrameUNet(num_frames=num_frames,
                                  combine_fn=combine_fn,
                                  num_layers=num_layers,
                                  features_start=features_start,
                                  bilinear=bilinear)



    def determine_loss_fn(self):
        if self.loss == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.loss == "ssim":
            self.criterion = SSIM()
        elif self.loss == "perceptual":
            self.criterion = Perceptual()
        elif self.loss == "l1_perceptual":
            self.criterion = L1_Perceptual()
        else:
            print("Using MSE Loss")
            self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, target, extra_info = batch
        pred = self(img)
        loss_val = F.mse_loss(pred.squeeze(), target.squeeze())
        self.log('train_loss', loss_val)

        # log images
        if self.hparams.log_tb_imgs and self.global_step % self.hparams.tb_img_freq == 0:
            self._log_images(img, target, pred, extra_info, step_name='train')

        # metrics
        ssim_val = ssim(pred, target)
        psnr_val = psnr(pred, target)
        self.log('train_ssim', ssim_val)
        self.log('train_psnr', psnr_val)

        return loss_val

    def validation_step(self, batch, batch_idx):
        img, target, extra_info = batch
        pred = self(img)
        loss_val = F.mse_loss(pred.squeeze(), target.squeeze())
        self.log('valid_loss', loss_val)

        # log predicted images
        if self.hparams.log_tb_imgs and self.global_step % self.hparams.tb_img_freq == 0:
            self._log_images(img, target, pred, extra_info, step_name='valid')

        # metrics
        ssim_val = ssim(pred, target)
        psnr_val = psnr(pred, target)
        self.log('valid_ssim', ssim_val)
        self.log('valid_psnr', psnr_val)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [opt]

    def _log_images(self, img, target, pred, extra_info, step_name, limit=1):
        # TODO: Randomly select image from batch instead of first image?
        img = img[:limit].squeeze(0)
        target = target[:limit].squeeze(0)
        pred = pred[:limit].squeeze(0)
        folder_name = extra_info['image_set'][0]
        frame_nums = extra_info['frame_nums'][0]

        if self.total_num_frames > 1:
            if self.is_color_input:
                img = img.reshape(self.total_num_frames, 3, img.shape[1], img.shape[2])
            else:
                img = img.reshape(self.total_num_frames, 1, img.shape[1], img.shape[2])
        img = torchvision.utils.make_grid(img, nrow=1)

        self.logger.experiment.add_image(f'{step_name}_input_images', img, self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_target', target, self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_pred', pred, self.trainer.global_step)
        self.logger.experiment.add_text(f'{step_name}_img_folder_path', f'{folder_name}: {frame_nums}', self.trainer.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, help="path to davinci data")
        parser.add_argument("--frames_per_sample", type=int, default=1, help="number of frames to include in each sample")
        parser.add_argument("--frames_to_drop", type=int, default=0, help="number of frames to randomly drop in each sample")
        parser.add_argument("--include_right_view", action='store_true', default=False, help="include left and right view")
        parser.add_argument("--stack_horizontal", action='store_true', default=False, help="stacks input views horizontally")
        parser.add_argument("--is_color_input", action='store_true', default=True, help="use color inputs instead of bw")
        parser.add_argument("--is_color_output", action='store_true', default=True, help="use color outputs instead of bw")
        parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
        parser.add_argument("--log_tb_imgs", action='store_true', default=False)
        parser.add_argument("--tb_img_freq", type=int, default=10000)
        parser.add_argument("--save_img_freq", type=int, default=50)
        parser.add_argument("--fid_freq", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument("--bilinear", action='store_true', default=False,
                            help="whether to use bilinear interpolation or transposed")

        return parser
