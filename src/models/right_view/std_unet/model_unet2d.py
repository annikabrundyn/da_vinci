import torch
import torch.nn.functional as F
import torchvision

import os.path
import numpy as np

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import ssim, psnr
import lpips

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from data.right_data import RightDaVinciDataModule
from models.unet.unet_plain import UNet

from metrics import FIDCallback


class UNet2DModel(pl.LightningModule):
    def __init__(
            self,
            num_frames: int,
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
        self.save_hyperparameters()
        self.num_frames = num_frames
        self.loss = loss

        self.criterion = self._determine_loss_fn()
        self.input_channels = 3 * self.num_frames

        self.net = UNet(
            input_channels = self.input_channels,
            output_channels = 3,
            num_layers = self.hparams.num_layers,
            features_start = self.hparams.features_start,
            bilinear = self.hparams.bilinear)

        self.LPIPS = lpips.LPIPS(net='alex')

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)

        loss_val = self.criterion(pred, target)
        ssim_val = ssim(pred, target)
        psnr_val = psnr(pred, target)
        lpips_val = self.LPIPS(pred, target)

        # TODO: is there a clean way to do this - not logging train metrics atm
        #logs = {'train_loss': loss_val, 'train_ssim': ssim_val, 'train_psnr': psnr_val}
        self.log('train_loss', loss_val)
        self.log('train_ssim', ssim_val)
        self.log('train_psnr', psnr_val)
        self.log('train_lpips', lpips_val)

        return loss_val

    def validation_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)

        loss_val = self.criterion(pred, target)
        ssim_val = ssim(pred, target)
        psnr_val = psnr(pred, target)
        lpips_val = self.LPIPS(pred, target)

        # TODO: find cleaner way
        # logs = {'val_loss': loss_val, 'val_ssim': ssim_val, 'val_psnr': psnr_val}
        self.log('val_loss', loss_val)
        self.log('val_ssim', ssim_val)
        self.log('val_psnr', psnr_val)
        self.log('val_lpips', lpips_val)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [opt]

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


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Required arguments
        parser.add_argument("--data_dir", type=str, help="path to davinci data")
        parser.add_argument("--num_frames", type=int, help="number of consecutive frames to include")
        parser.add_argument("--loss", type=str, choices=['l1', 'mse', 'ssim', 'perceptual', 'l1_perceptual', 'L1_SSIM'], help="loss function")
        parser.add_argument("--bilinear", action='store_true', help="bilinear (True) vs. transposed convolution (False)")
        parser.add_argument("--num_layers", type=int, help="number of layers/blocks in u-net")

        # hyperparameters with a default value
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument("--num_workers", type=int, default=8)

        # logging
        parser.add_argument("--log_tb_imgs", action='store_true', default=False)
        parser.add_argument("--tb_img_freq", type=int, default=10000)
        parser.add_argument("--save_img_freq", type=int, default=50)
        parser.add_argument("--fid_freq", type=int, default=500)
        return parser


if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    print("start right multiframe model")
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = UNet2DModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = RightDaVinciDataModule(
        args.data_dir,
        frames_per_sample=args.num_frames,
        frames_to_drop=0,
        is_color_input=True,
        is_color_output=True,
        extra_info=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()
    print("dm setup")

    # sanity check
    print("size of trainset:", len(dm.train_samples))
    print("size of validset:", len(dm.val_samples))
    print("size of testset:", len(dm.test_samples))

    img, target = next(iter(dm.train_dataloader()))
    print(img.shape)
    print(target.shape)

    # model
    model = UNet2DModel(**args.__dict__)
    print("model instance created")
    print("lightning version", pl.__version__)

    # train
    #trainer = pl.Trainer.from_argparse_args(args, callbacks=[FIDCallback("real_stats.pickle", dm, num_samples=5)])
    trainer = pl.Trainer.from_argparse_args(args)
    print("trainer created")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())