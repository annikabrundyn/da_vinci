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


class LeftRightStackedDM(DepthMap):

    def __init__(self):
        super().__init__()

        self.net = UNet()

    def training_step(self, batch, batch_idx):
        left, right, target = batch
        input_imgs = torch.cat((left, right))
        pred = self(input_imgs)


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
    model = DepthMap(**args.__dict__)

    # train
    trainer = pl.Trainer().from_argparse_args(args)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
