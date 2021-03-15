from argparse import ArgumentParser

import torch
from torchvision.utils import make_grid

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import ssim, psnr
import lpips
import numpy as np

from losses import Perceptual, L1_Perceptual, L1_SSIM, SSIM

#from models.unet_architecture import MultiFrameUNet


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        num_frames: int,
        combine_fn: str,
        loss: str,
        extra_skip: str,
        num_layers: int,
        bilinear: str,
        sigmoid_on_output: bool,
        features_start: int = 64,
        lr: float = 0.001,
        log_tb_imgs: bool = False,
        tb_img_freq: int = 8000,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_frames = num_frames
        self.combine_fn = combine_fn
        self.loss = loss
        self.bilinear = (bilinear in ("True", "T", "true"))

        self.criterion = self._determine_loss_fn()

        self.LPIPS = lpips.LPIPS(net='alex')

        # NOTE: for all the different models, this is essentially the only thing that changes (architecture)
        # self.net = MultiFrameUNet(num_frames=num_frames,
        #                           combine_fn=combine_fn,
        #                           num_layers=num_layers,
        #                           features_start=features_start,
        #                           bilinear=bilinear)

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

    def _calculate_loss_metrics(self, target, pred, step_name):
        # calculate loss
        loss_val = self.criterion(pred, target)

        # calculate metrics
        ssim_val = ssim(pred, target.type(pred.dtype))
        psnr_val = psnr(pred, target)
        lpips_val = self.LPIPS(pred, target).mean()

        # return loss and metric values to log
        logs = {f'{step_name}_loss': loss_val, f'{step_name}_ssim': ssim_val, f'{step_name}_psnr': psnr_val,
                f'{step_name}_lpips': lpips_val}

        return logs

    def training_step(self, batch, batch_idx):
        # predict right view
        img, target = batch
        pred = self(img)

        # returns a dictionary with loss and metrics
        logs = self._calculate_loss_metrics(target, pred, "train")

        # log metrics to tensorboard
        self.log_dict(logs)

        # log predicted images every 10k steps
        if self.hparams.log_tb_imgs and self.global_step % self.hparams.tb_img_freq == 0:
            # pick random element in batch to visualize (train dataloader is shuffled)
            self._log_images(img[0], target[0], pred[0], step_name="train")

        return logs['train_loss']

    def validation_step(self, batch, batch_idx):
        # predict right view
        img, target = batch
        pred = self(img)

        # returns a dictionary with loss and metrics
        logs = self._calculate_loss_metrics(target, pred, "val")

        # log metrics to tensorboard
        self.log_dict(logs)

        # log predicted images - already saving same val images - not logging to tb [new change: 03/11]
        # if self.hparams.log_tb_imgs and self.global_step % self.hparams.tb_batch_freq == 0:
        #     # pick random element in batch to visualize - val dataloader is not shuffled
        #     idx = np.random.choice(len(img))
        #     self._log_images(img[idx], target[idx], pred[idx], step_name="val")

    def test_step(self, batch, batch_idx):
        # predict right view
        img, target = batch
        pred = self(img)

        logs = self._calculate_loss_metrics(target, pred, "test")
        return logs

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [opt]

    def _log_images(self, img, target, pred, step_name):

        # first frame is latest --> flip so that first frame is earliest and last frame is latest in visualization
        # dim 0 is frame axis
        img = img.flip(dims=[0])

        self.logger.experiment.add_image(f'{step_name}_input_left', make_grid(img), self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_target_right', make_grid(target), self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_pred_right', make_grid(pred), self.trainer.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Required arguments
        parser.add_argument("--data_dir", type=str, help="path to davinci data folder")
        parser.add_argument("--num_frames", type=int, default=5, help="number of consecutive frames per sample")
        parser.add_argument("--combine_fn", type=str, default='conv3d', choices=['conv3d', 'max', 'average'],
                            help="how to combine multiple frames")
        parser.add_argument("--loss", type=str, default='mse', choices=['l1', 'mse', 'ssim', 'perceptual', 'l1_perceptual', 'l1_ssim'],
                            help="loss function")
        parser.add_argument("--extra_skip", type=str, default='False', help="whether to add extra skip connection from input to output")
        parser.add_argument("--bilinear", type=str, default='False', help="bilinear upsampling ('True', 'T', 'true') vs. transposed convolution")
        parser.add_argument("--sigmoid_on_output", action='store_true', default=False, help="apply sigmoid on output")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers/blocks in u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")

        # hyperparameters with a default value
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate for adam optimizer")
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--num_workers", type=int, default=4)

        # logging
        parser.add_argument("--ckpt_path", type=str, default=None, help="load checkpoint provided at this path")
        parser.add_argument("--log_tb_imgs", action='store_true', default=False)
        parser.add_argument("--tb_step_freq", type=int, default=10000, help="log image to tensborboard every x steps")
        parser.add_argument("--save_epoch_freq", type=int, default=10)
        parser.add_argument("--fid_epoch_freq", type=int, default=5, help="number of epochs between each fid calculation")
        parser.add_argument("--fid_n_samples", type=int, default=10000, help="number of samples to use in fid")

        return parser

