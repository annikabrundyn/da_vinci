import torch
import torch.nn.functional as F
import torchvision

import os.path
import numpy as np

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import ssim, psnr
from pytorch_lightning.metrics.regression import SSIM

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from data.right_data import RightDaVinciDataModule
from models.right_view.unet_add_connect import ModifiedUNet
from losses.perceptual_loss import Perceptual


class ModifiedUNetModel(pl.LightningModule):
    def __init__(
        self,
        frames_per_sample: int = 1,
        frames_to_drop: int = 0,
        is_color_input: bool = True,
        is_color_output: bool = True,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
        loss: str = 'ssim',
        lr: float = 0.001,
        log_tb_imgs: bool = False,
        tb_img_freq: int = 10000,
        **kwargs
    ):
        super().__init__()

        self.frames_per_sample = frames_per_sample
        self.frames_to_drop = frames_to_drop
        self.is_color_input = is_color_input
        self.is_color_output = is_color_output
        self.loss = loss

        self.save_hyperparameters()

        self._calc_input_channels()

        if is_color_output:
            self.num_classes = 3
        else:
            self.num_classes = 1

        self.net = ModifiedUNet(num_classes=self.num_classes,
                             input_channels=self.input_channels,
                             num_layers=self.hparams.num_layers,
                             features_start=self.hparams.features_start,
                             bilinear=self.hparams.bilinear)

        if self.loss == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.loss == "ssim":
            self.criterion = SSIM()
        elif self.loss == "perceptual":
            self.criterion = Perceptual()
        else:
            self.criterion = torch.nn.MSELoss()


    def _calc_input_channels(self):
        # calculate the input channels for UNet

        if self.frames_per_sample <= 2:
            self.input_channels = self.frames_per_sample
        else:
            max_drop_frames = self.frames_per_sample - 2
            self.frames_to_drop = min(self.frames_to_drop, max_drop_frames)
            self.input_channels = self.frames_per_sample - self.frames_to_drop

        self.total_num_frames = self.input_channels

        if self.is_color_input:
            self.input_channels = 3 * self.input_channels

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, target, extra_info = batch
        _, pred = self(img)
        loss_val = self.criterion(pred.squeeze(), target.squeeze())
        #loss_val = F.mse_loss(pred.squeeze(), target.squeeze())
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
        _, pred = self(img)
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



if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    print("start right direct model")
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = ModifiedUNetModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = RightDaVinciDataModule(
        args.data_dir,
        frames_per_sample=args.frames_per_sample,
        frames_to_drop=args.frames_to_drop,
        is_color_input=args.is_color_input,
        is_color_output=args.is_color_output,
        extra_info=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()
    print("dm setup")

    # sanity check
    print("size of trainset:", len(dm.train_samples))
    print("size of validset:", len(dm.val_samples))
    print("size of testset:", len(dm.test_samples))

    img, target, extra_info = next(iter(dm.train_dataloader()))
    print(img.shape)
    print(target.shape)
    print(len(extra_info))

    # model
    model = ModifiedUNetModel(**args.__dict__)
    print("model instance created")
    print("lightning version", pl.__version__)

    # train
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=[RightCallback(args.save_img_freq)])
    trainer = pl.Trainer.from_argparse_args(args)
    print("trainer created")
    trainer.fit(model, dm)