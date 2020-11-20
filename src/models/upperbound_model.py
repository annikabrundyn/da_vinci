from argparse import ArgumentParser

import os.path
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.unet import UNet
from metrics.fid import calculate_fid
from data.data import DaVinciDataModule

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from pytorch_lightning.metrics.functional import ssim, psnr
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback

from models.model import SavePredImgCallback
from models.model import DepthMap


class UpperBoundImgCallback(Callback):

    def __init__(self, dl, epoch_logging_freq: int = 50):
        # save predictions every 50 epochs
        self.epoch_logging_freq = epoch_logging_freq
        self.dl = dl

    def on_sanity_check_end(self, trainer, pl_module):
        '''save the input + target images only once'''
        print("do this only once")
        batch_idx = 0
        for img, target, extra in self.dl:
            folder_name = extra['image_set'][0]
            frame_nums = extra['frame_nums'][0]

            dir = trainer.checkpoint_callback.dirpath
            dir = os.path.split(dir)[0]
            inputs_dir_path = os.path.join(dir, "inputs")
            targets_dir_path = os.path.join(dir, "targets")

            pl_module._matplotlib_imshow_input_imgs(img.squeeze(0), folder_name, frame_nums, save_fig=True, title=f"input_{batch_idx}", dir_path=inputs_dir_path)
            pl_module._matplotlib_imshow_dm(target.squeeze(0), title=f"target_{batch_idx}", save_fig=True, dir_path=targets_dir_path)

            batch_idx += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        '''save only the predicted '''
        print('a')
        if trainer.current_epoch % self.epoch_logging_freq == 0:
            batch_idx = 0
            for img, target, extra in self.dl:
                img, target = img.to(pl_module.device), target.to(pl_module.device)
                folder_name = extra['image_set'][0]
                frame_nums = extra['frame_nums'][0]

                pred = pl_module(img)

                dir = trainer.checkpoint_callback.dirpath
                dir = os.path.split(dir)[0]
                dir_path = os.path.join(dir, f"epoch_{trainer.current_epoch}", "pred")

                pl_module._matplotlib_imshow_dm(pred.squeeze(0), title=f"prediction_{batch_idx}", save_fig=True, dir_path=dir_path)

                batch_idx += 1


class UpperBoundModel(DepthMap):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.include_right_view = True

        self._calc_input_channels()

        self.net = UNet(num_classes=self.hparams.num_classes,
                        input_channels=self.input_channels,
                        num_layers=self.hparams.num_layers,
                        features_start=self.hparams.features_start,
                        bilinear=self.hparams.bilinear)

    def _calc_input_channels(self):

        if self.frames_per_sample == 1 and self.frames_to_drop == 0:
            self.input_channels = 1

            if self.include_right_view:
                self.input_channels = 2 * self.input_channels

            if self.is_color_input:
                self.input_channels = 3 * self.input_channels

        else:
            raise ValueError("This model currently only works with single frames (color or b/w)")

    def training_step(self, batch, batch_idx):
        img, target, extra_info = batch
        pred = self(img)
        loss_val = F.mse_loss(pred.squeeze(), target.squeeze())
        self.log('train_loss', loss_val)

        # log images
        if self.global_step % self.hparams.output_img_freq == 0:
            self._log_images(img, target, pred, extra_info, step_name='train')

        # metrics
        ssim_val = ssim(pred, target)
        psnr_val = psnr(pred, target)
        fid_val = calculate_fid(pred, target)
        self.log('train_ssim', ssim_val)
        self.log('train_psnr', psnr_val)
        self.log('train_fid', fid_val)
        
        return loss_val

    def validation_step(self, batch, batch_idx):
        img, target, extra_info = batch
        pred = self(img)
        loss_val = F.mse_loss(pred.squeeze(), target.squeeze())
        self.log('valid_loss', loss_val)

        # log predicted images
        if self.global_step % self.hparams.output_img_freq == 0:
            self._log_images(img, target, pred, extra_info, step_name='valid')

        # metrics
        ssim_val = ssim(pred, target)
        psnr_val = psnr(pred, target)
        fid_val = calculate_fid(pred, target)
        self.log('valid_ssim', ssim_val)
        self.log('valid_psnr', psnr_val)
        self.log('valid_fid', fid_val)

    def _matplotlib_imshow_input_imgs(self, img, folder_name, frame_nums, save_fig=False, title=None, dir_path=None):
        """Summary

        Args:
            img (tensor): (num_channels, H, W)

        Returns:
            TYPE: fig
        """

        nrow = 1
        ncol = 2

        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(nrow, ncol),
                         direction='column',
                         axes_pad=0.3)

        if self.is_color_input:
            range_inputs = range(self.input_channels // 3)
        else:
            range_inputs = range(self.input_channels)

        for ax, idx, side in zip(grid, range_inputs, ['left', 'right']):
            if self.is_color_input:
                # select 3 channels for color inputs
                npimg = img[3*idx:3*(idx+1)].squeeze().detach().cpu().numpy()
                npimg = np.transpose(npimg, (1, 2, 0))
                ax.imshow(npimg)
            else:
                npimg = img[idx].squeeze().detach().cpu().numpy()
                ax.imshow(npimg, cmap='gray')
            ax.axis('off')
            ax.set_title(f"{side} view: {folder_name}/{frame_nums}")

        if save_fig:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            img_path = os.path.join(dir_path, f"{title}.png")
            plt.savefig(img_path, bbox_inches='tight')
            plt.close()

        return fig



if __name__ == '__main__':
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    print("start")
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = UpperBoundModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = DaVinciDataModule(args.data_dir,
                           frames_per_sample=args.frames_per_sample,
                           frames_to_drop=args.frames_to_drop,
                           include_right_view=True,
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
    model = UpperBoundModel(**args.__dict__)
    print("model instance created")
    print('lightning version', pl.__version__)

    # train
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[UpperBoundImgCallback(dm.vis_img_dataloader())])
    print("trainer created")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
