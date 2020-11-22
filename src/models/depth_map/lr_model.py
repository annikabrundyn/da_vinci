from argparse import ArgumentParser

import pytorch_lightning as pl

from models.depth_map.unet import UNet
from data.data import DaVinciDataModule
from models.depth_model import DepthMap


class LeftRightDepthMap(DepthMap):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.net = UNet(num_classes=self.num_classes,
                        input_channels=2*self.input_channels,
                        num_layers=self.num_layers,
                        features_start=self.features_start,
                        bilinear=self.bilinear)

    def _log_images(self, *args, **kwargs):
        super()._log_images(*args, **kwargs, nrow=self.input_channels)


if __name__ == '__main__':
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = LeftRightDepthMap.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = DaVinciDataModule(args.data_dir,
                           frames_per_sample=args.frames_per_sample,
                           frames_to_drop=args.frames_to_drop,
                           include_right_view=True,
                           extra_info=True,
                           batch_size=args.batch_size)
    # have to run setup to do splits/sliding window/etc
    dm.setup()

    # sanity check
    print("size of trainset:", len(dm.train_samples))
    print("size of validset:", len(dm.val_samples))
    print("size of testset:", len(dm.test_samples))

    # model
    model = LeftRightDepthMap(**args.__dict__)

    # train
    trainer = pl.Trainer().from_argparse_args(args)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
