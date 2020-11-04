from model import DepthMap

from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision

from unet_depth import UNetDepth
from data_v2 import DaVinciDataModule

import numpy as np
import matplotlib.pyplot as plt

from pytorch_lightning.metrics.functional import ssim, psnr


class DirectUNet(DepthMap):
    def __init__(
        self,
        frames_per_sample: int,
        frames_to_drop: int,
        num_classes: int = 3,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
        lr: float = 0.001,
        output_img_freq: int = 100,
        batch_size: int = 16,
        **kwargs,
    ):

        super().__init__(frames_per_sample,
        frames_to_drop,
        num_classes,
        num_layers,
        features_start,
        bilinear,
        lr,
        output_img_freq,
        batch_size,
        **kwargs)

        self.net = UNetDirect(
            num_classes=num_classes,
            input_channels=self.input_channels,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )

    def _calc_input_channels(self):
        # calculate the input channels for UNetDepth
        if self.frames_per_sample <= 2:
            self.input_channels = self.frames_per_sample * self.num_classes
        else:
            max_drop_frames = self.frames_per_sample - 2
            self.frames_to_drop = min(self.frames_to_drop, max_drop_frames)
            self.input_channels = (self.frames_per_sample - self.frames_to_drop) * self.num_classes



    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, help="path to davinci data")
        parser.add_argument(
            "--frames_per_sample",
            type=int,
            help="number of frames to include in each sample",
        )
        parser.add_argument(
            "--frames_to_drop",
            type=int,
            help="number of frames to randomly drop in each sample",
        )
        parser.add_argument(
            "--num_classes", type=int, default=3, help="output channels"
        )
        parser.add_argument(
            "--batch_size", type=int, default=16, help="size of the batches"
        )
        parser.add_argument("--output_img_freq", type=int, default=100)
        parser.add_argument(
            "--lr", type=float, default=0.001, help="adam: learning rate"
        )
        parser.add_argument(
            "--num_layers", type=int, default=5, help="number of layers on u-net"
        )
        parser.add_argument(
            "--features_start",
            type=float,
            default=64,
            help="number of features in first layer",
        )
        parser.add_argument(
            "--bilinear",
            action="store_true",
            default=False,
            help="whether to use bilinear interpolation or transposed",
        )

        return parser


if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = DepthMap.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = DirectDaVinciDataModule(
        args.data_dir,
        frames_per_sample=args.frames_per_sample,
        frames_to_drop=args.frames_to_drop,
        batch_size=args.batch_size,
    )
    dm.setup()

    # sanity check
    print("size of trainset:", len(dm.train_dataset))
    print("size of validset:", len(dm.val_dataset))
    print("size of testset:", len(dm.test_dataset))

    # model
    model = DepthMap(**args.__dict__)

    # train
    trainer = pl.Trainer().from_argparse_args(args)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
