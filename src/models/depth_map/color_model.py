from argparse import ArgumentParser

import os.path
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.depth_map.base_model import BaseDepthMap
from models.depth_map.unet import UNet
from metrics.fid import calculate_fid
from data.data import DaVinciDataModule
from models.callbacks.img_save import SaveImgCallBack

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from pytorch_lightning.metrics.functional import ssim, psnr


class ColorModel(BaseDepthMap):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_color_input = True

        self._calc_input_channels()

        self.net = UNet(num_classes=self.hparams.num_classes,
                        input_channels=self.input_channels,
                        num_layers=self.hparams.num_layers,
                        features_start=self.hparams.features_start,
                        bilinear=self.hparams.bilinear)

    def _calc_input_channels(self):
        # calculate the input channels for UNet
        if self.frames_per_sample <= 2:
            self.input_channels = self.frames_per_sample
        else:
            max_drop_frames = self.frames_per_sample - 2
            self.frames_to_drop = min(self.frames_to_drop, max_drop_frames)
            self.input_channels = self.frames_per_sample - self.frames_to_drop

        self.input_channels = 3 * self.input_channels

    def _matplotlib_imshow_input_imgs(self, img, folder_name, frame_nums, save_fig=False, title=None, dir_path=None):

        nrow = self.input_channels // 3
        ncol = 1

        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(nrow, ncol),
                         direction='column',
                         axes_pad=0.3)

        for ax, idx in zip(grid, range(nrow)):
            # select 3 channels for color inputs
            npimg = img[3*idx:3*(idx+1)].squeeze().detach().cpu().numpy()
            npimg = np.transpose(npimg, (1, 2, 0))
            ax.imshow(npimg)
            ax.axis('off')
            ax.set_title(f"{folder_name}/{frame_nums}")

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
    parser = ColorModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = DaVinciDataModule(args.data_dir,
                           frames_per_sample=args.frames_per_sample,
                           frames_to_drop=args.frames_to_drop,
                           include_right_view=False,
                           is_color_input=True,
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
    model = ColorModel(**args.__dict__)
    print("model instance created")
    print('lightning version', pl.__version__)

    # train
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[SaveImgCallBack(dm.vis_img_dataloader())])
    print("trainer created")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
