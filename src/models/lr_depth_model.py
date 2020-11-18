from argparse import ArgumentParser

import pytorch_lightning as pl

from models.unet import UNet
from data.data import DaVinciDataModule
from models.model import Model


class LeftRightDepthMap(Model):
    def __init__(
            self,
            frames_per_sample: int,
            frames_to_drop: int,
            include_right_view: bool = True,
            stack_horizontal: bool = False,
            is_color_input: bool = False,
            is_color_output: bool = False,
            num_classes: int = 1,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            lr: float = 0.001,
            output_img_freq : int = 500,
            fid_freq : int = 500,
            **kwargs
    ):
        super().__init__(frames_per_sample,
                         frames_to_drop,
                         include_right_view,
                         stack_horizontal,
                         is_color_input,
                         is_color_output,
                         num_classes,
                         num_layers,
                         features_start,
                         bilinear,
                         lr,
                         output_img_freq,
                         fid_freq,
                         **kwargs)


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
