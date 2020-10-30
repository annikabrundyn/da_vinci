from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision

from unet import UNet
from data import DaVinciDataModule
from model import DepthMap

import numpy as np
import matplotlib.pyplot as plt

from pytorch_lightning.metrics.functional import ssim, psnr


class LeftRightDepthMap(DepthMap):

    def __init__(
            self,
            lr: float = 0.001,
            num_classes: int = 1,
            frames_per_sample: int = 5,
            frames_to_drop: int = 2,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            output_img_freq: int = 100,
            batch_size: int = 16,
            **kwargs
    ):
        super().__init__()

        self.net = UNet(num_classes=num_classes,
                        input_channels=2*self.input_channels,
                        num_layers=self.num_layers,
                        features_start=self.features_start,
                        bilinear=self.bilinear)

    def _log_images(self, img, target, pred, step_name, limit=1):
        # TODO: Randomly select image from batch
        img = img[:limit]
        target = target[:limit]
        pred = pred[:limit]

        # Log input/original image
        img = img.permute(1, 0, 2, 3)
        input_images = torchvision.utils.make_grid(img, nrow=self.input_channels)
        self.logger.experiment.add_image(f'{step_name}_input_img', input_images, self.trainer.global_step)

        # Log colorized depth maps - using magma colormap
        color_target_dm = self._matplotlib_imshow(target)
        color_pred_dm = self._matplotlib_imshow(pred)

        self.logger.experiment.add_figure(f'{step_name}_target_dm_color', color_target_dm, self.trainer.global_step)
        self.logger.experiment.add_figure(f'{step_name}_pred_dm_color', color_pred_dm, self.trainer.global_step)


if __name__ == '__main__':
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
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
                           include_right_view=True,
                           batch_size=args.batch_size)

    # sanity check
    print("size of trainset:", len(dm.train_dataset))
    print("size of validset:", len(dm.val_dataset))
    print("size of testset:", len(dm.test_dataset))

    # model
    model = LeftRightDepthMap(**args.__dict__)
    #model._log_images()

    # train
    trainer = pl.Trainer().from_argparse_args(args)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
