from argparse import ArgumentParser

import os.path
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.depth_map.unet import UNet
from metrics.fid import calculate_fid
from data.data import DaVinciDataModule

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from pytorch_lightning.metrics.functional import ssim, psnr
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback


class SavePredImgCallback(Callback):
    def __init__(self, dl, epoch_logging_freq: int = 50):
        # save every 50 epochs
        self.epoch_logging_freq = epoch_logging_freq
        self.dl = dl

    def on_validation_epoch_end(self, trainer, pl_module):
        print('a')
        if trainer.current_epoch % self.epoch_logging_freq == 0:
            batch_idx = 0
            for img, target, extra in self.dl:
                img, target = img.to(pl_module.device), target.to(pl_module.device)
                folder_name = extra_info['image_set'][0]
                frame_nums = extra_info['frame_nums'][0]

                pred = pl_module(img)

                pl_module._matplotlib_imshow_input_imgs(img.squeeze(0), folder_name, frame_nums, save_fig=True, title=f"input_{batch_idx}")
                pl_module._matplotlib_imshow_dm(target.squeeze(0), title=f"target_{batch_idx}", save_fig=True, location="target")
                pl_module._matplotlib_imshow_dm(pred.squeeze(0), title=f"prediction_{batch_idx}", save_fig=True, location="pred")

                batch_idx += 1


class DepthMap(pl.LightningModule):
    def __init__(
            self,
            frames_per_sample: int,
            frames_to_drop: int,
            include_right_view: bool = False,
            stack_horizontal: bool = False,
            is_color_input: bool = False,
            num_classes: int = 1,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            lr: float = 0.001,
            output_img_freq : int = 5000,
            fid_freq : int = 500,
            **kwargs
    ):
        super().__init__()

        self.frames_per_sample = frames_per_sample
        self.frames_to_drop = frames_to_drop
        self.include_right_view = include_right_view
        self.stack_horizontal = stack_horizontal
        self.is_color_input = is_color_input

        self.save_hyperparameters()

        self._calc_input_channels()

        if self.stack_horizontal:
            if self.include_right_view:
                num_stack_horizontal = (self.frames_per_sample - self.frames_to_drop) * 2
            else:
                num_stack_horizontal = (self.frames_per_sample - self.frames_to_drop)
        else:
            num_stack_horizontal = 1

        self.net = UNet(num_classes=self.hparams.num_classes,
                        input_channels=self.input_channels,
                        num_stack_horizontal=num_stack_horizontal,
                        num_layers=self.hparams.num_layers,
                        features_start=self.hparams.features_start,
                        bilinear=self.hparams.bilinear)


    def _calc_input_channels(self):
        # calculate the input channels for UNet
        if self.stack_horizontal:
            self.input_channels=1
        elif not self.stack_horizontal:
            if self.frames_per_sample <= 2:
                self.input_channels = self.frames_per_sample
            else:
                max_drop_frames = self.frames_per_sample - 2
                self.frames_to_drop = min(self.frames_to_drop, max_drop_frames)
                self.input_channels = self.frames_per_sample - self.frames_to_drop

            if self.include_right_view:
                self.input_channels = 2 * self.input_channels

        if self.is_color_input:
            self.input_channels = self.input_channels * 3

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, target, extra_info = batch
        pred = self(img)
        loss_val = F.mse_loss(pred.squeeze(), target.squeeze())
        self.log('train_loss', loss_val)

        # log images
        if self.global_step % self.hparams.output_img_freq == 0:
            self._log_images(img, target, pred, extra_info, step_name='train')

        # log fid
        if self.global_step % self.hparams.fid_freq == 0:
            self._log_fid(pred, target, step_name='train')

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
        if self.global_step % self.hparams.output_img_freq == 0:
            self._log_images(img, target, pred, extra_info, step_name='valid')

        # log FID
        if self.global_step % self.hparams.fid_freq == 0:
            self._log_fid(pred, target, step_name='valid')

        # metrics
        ssim_val = ssim(pred, target)
        psnr_val = psnr(pred, target)
        self.log('valid_ssim', ssim_val)
        self.log('valid_psnr', psnr_val)

    # def test_step(self, batch, batch_idx):
    #     # batch size is 1 in the validation pred images
    #     img, target, extra_info = batch
    #     folder_name = extra_info['image_set'][0]
    #     frame_nums = extra_info['frame_nums'][0]
    #
    #     pred = self(img)
    #
    #     self._matplotlib_imshow_input_imgs(img.squeeze(0), folder_name, frame_nums, save_fig=True, title=f"input_{batch_idx}")
    #     self._matplotlib_imshow_dm(target.squeeze(0), title=f"target_{batch_idx}", save_fig=True)
    #     self._matplotlib_imshow_dm(pred.squeeze(0), title=f"prediction_{batch_idx}", save_fig=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt]

    def _matplotlib_imshow_input_imgs(self, img, folder_name, frame_nums, save_fig=False, title=None):
        """Summary

        Args:
            img (tensor): (num_channels, H, W)

        Returns:
            TYPE: fig
        """
        if self.stack_horizontal:
            nrow = self.input_channels
            ncol = 1
        elif not self.stack_horizontal:
            if self.include_right_view:
                nrow = self.input_channels // 2
                ncol = 2
            else:
                nrow = self.input_channels
                ncol = 1

        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(nrow, ncol),
                         direction='column',
                         axes_pad=0.3)

        if self.is_color_input:
            range_inputs = range(self.input_channels // 3)
        else:
            range_inputs = range(self.input_channels)

        for ax, idx in zip(grid, range_inputs):
            if self.is_color_input:
                # select 3 channels for color inputs
                npimg = img[3*idx:3*(idx+1)].squeeze().detach().cpu().numpy()
                npimg = np.transpose(npimg, (1, 2, 0))
                ax.imshow(npimg)
            else:
                npimg = img[idx].squeeze().detach().cpu().numpy()
                ax.imshow(npimg, cmap='gray')
            ax.axis('off')
            side = 'left'
            if idx >= nrow:
                side = 'right'
                idx = idx - nrow
            ax.set_title(f"{side} view: {folder_name}/{frame_nums[idx]}/include_right_view:{self.include_right_view}")

        if save_fig:
            dir = trainer.checkpoint_callback.dirpath
            dir = os.path.split(dir)[0]
            dir_path = os.path.join(dir, f"epoch_{self.current_epoch}", "input")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            img_path = os.path.join(dir_path, f"{title}.png")
            plt.savefig(img_path, bbox_inches='tight')
            plt.close()

        return fig

    def _matplotlib_imshow_dm(self, img, title, inverse=True, cmap='magma', save_fig=False, dir_path=None):
        if inverse:
            img = 1 - img
        npimg = img.squeeze().detach().cpu().numpy()
        fig = plt.figure()
        plt.imshow(npimg, cmap=cmap)
        plt.title(title)

        if save_fig:
            #dir_path = os.path.join(trainer.log_dir, f"epoch_{self.current_epoch}", location)

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            img_path = os.path.join(dir_path, f"{title}.png")
            plt.savefig(img_path, bbox_inches='tight')
            plt.close()

        return fig

    def _log_images(self, img, target, pred, extra_info, step_name, limit=1):
        # TODO: Randomly select image from batch instead of first image?
        img = img[:limit]
        target = target[:limit]
        pred = pred[:limit]
        folder_name = extra_info['image_set'][0]
        frame_nums = extra_info['frame_nums'][0]
        frame_nums = frame_nums.split()

        fig = self._matplotlib_imshow_input_imgs(img.squeeze(0), folder_name, frame_nums)
        self.logger.experiment.add_figure(f'{step_name}_input_images', fig, self.trainer.global_step)

        # Log colorized depth maps - using magma colormap
        color_target_dm = self._matplotlib_imshow_dm(target, "target")
        color_pred_dm = self._matplotlib_imshow_dm(pred, "prediction")

        self.logger.experiment.add_figure(f'{step_name}_target_dm_color', color_target_dm, self.trainer.global_step)
        self.logger.experiment.add_figure(f'{step_name}_pred_dm_color', color_pred_dm, self.trainer.global_step)

    def _log_fid(self, pred, target, step_name):
        fid_val = calculate_fid(pred, target, is_color_input=self.is_color_input, device=self.device)
        self.logger.experiment.add_scalar(f"{step_name}_fid", fid_val, self.trainer.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, help="path to davinci data")
        parser.add_argument("--frames_per_sample", type=int, default=1, help="number of frames to include in each sample")
        parser.add_argument("--frames_to_drop", type=int, default=0, help="number of frames to randomly drop in each sample")
        parser.add_argument("--include_right_view", action='store_true', default=False, help="include left and right view")
        parser.add_argument("--stack_horizontal", action='store_true', default=False, help="stacks input views horizontally")
        parser.add_argument("--is_color_input", action='store_true', default=False, help="use color inputs instead of bw")
        parser.add_argument("--num_classes", type=int, default=1, help="output channels")
        parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
        parser.add_argument("--output_img_freq", type=int, default=100)
        parser.add_argument("--fid_freq", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument("--bilinear", action='store_true', default=False,
                            help="whether to use bilinear interpolation or transposed")

        return parser


if __name__ == '__main__':
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    print("start")
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = DepthMap.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = DaVinciDataModule(args.data_dir,
                           frames_per_sample=args.frames_per_sample,
                           frames_to_drop=args.frames_to_drop,
                           include_right_view=args.include_right_view,
                           stack_horizontal=args.stack_horizontal,
                           is_color_input=args.is_color_input,
                           extra_info=True,
                           batch_size=args.batch_size,
                           num_workers=args.num_workers)
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
    model = DepthMap(**args.__dict__)
    print("model instance created")
    print('lightning version', pl.__version__)

    # train
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[SavePredImgCallback(dm.vis_img_dataloader())])
    print("trainer created")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    # predict + save val images
    # trainer.test(model, dm.vis_img_dataloader())
