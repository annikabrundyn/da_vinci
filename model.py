from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision

from unet import UNet
from data import DaVinciDataModule

import numpy as np
import matplotlib.pyplot as plt


class DepthMap(pl.LightningModule):
    def __init__(
            self,
            lr: float = 0.001,
            num_classes: int = 1,
            frames_per_sample: int = 5,
            frames_to_drop: int = 2,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            output_img_freq : int = 100,
            batch_size : int = 16,
            **kwargs
    ):

        super().__init__()

        self.num_classes = num_classes
        self.frames_per_sample = frames_per_sample
        self.frames_to_drop = frames_to_drop
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr
        self.output_img_freq = output_img_freq
        self.batch_size = batch_size

        self.net = UNet(num_classes=num_classes,
                        frames_per_sample=self.frames_per_sample,
                        frames_to_drop=self.frames_to_drop,
                        num_layers=self.num_layers,
                        features_start=self.features_start,
                        bilinear=self.bilinear)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        if batch_idx % self.output_img_freq == 0:
            self._log_images(img, target, pred, step_name='train')
        loss_val = F.mse_loss(pred.squeeze(), target.squeeze())
        self.log('train_loss', loss_val)
        return loss_val

    def validation_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        if batch_idx % self.output_img_freq == 0:
            self._log_images(img, target, pred, step_name='valid')
        loss_val = F.mse_loss(pred.squeeze(), target.squeeze())
        self.log('valid_loss', loss_val)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt]

    def _matplotlib_imshow(self, img, inverse=True, cmap='magma'):
        if inverse:
            img = 1 - img
        npimg = img.squeeze().detach().cpu().numpy()
        fig = plt.figure()
        plt.imshow(npimg, cmap=cmap)
        return fig

    def _log_images(self, img, target, pred, step_name, limit=1):
        # TODO: Randomly select image from batch
        img = img[:limit]
        target = target[:limit]
        pred = pred[:limit]

        # Log input/original image
        img = img.permute(1, 0, 2, 3)
        input_images = torchvision.utils.make_grid(img, nrow=1)
        self.logger.experiment.add_image(f'{step_name}_input_img', input_images, self.trainer.global_step)

        # Log colorized depth maps - using magma colormap
        color_target_dm = self._matplotlib_imshow(target)
        color_pred_dm = self._matplotlib_imshow(pred)

        self.logger.experiment.add_figure(f'{step_name}_target_dm_color', color_target_dm, self.trainer.global_step)
        self.logger.experiment.add_figure(f'{step_name}_pred_dm_color', color_pred_dm, self.trainer.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default='.', help="path to nyu depth data")
        parser.add_argument("--resize", type=float, default=1, help="percent to downsample images")
        parser.add_argument("--frames_per_sample", type=int, default=5, help="number of frames to include in each sample")
        parser.add_argument("--frames_to_drop", type=int, default=2, help="number of frames to randomly drop in each sample")
        parser.add_argument("--num_classes", type=int, default=1, help="output channels")
        parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
        parser.add_argument("--output_img_freq", type=int, default=100)
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument("--bilinear", action='store_true', default=False,
                            help="whether to use bilinear interpolation or transposed")

        return parser


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
                           resize=args.resize,
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
