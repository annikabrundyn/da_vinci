from argparse import ArgumentParser

import pytorch_lightning as pl
import lpips

from deprecated.right_data import RightDaVinciDataModule
from models.right_view.stacked_unet2d import StackedModel
from models.unet_architecture import UNetExtraSkip


class ExtraSkipModel(StackedModel):
    def __init__(
            self,
            num_frames: int,
            combine_fn: str,
            loss: str,
            num_layers: int,
            bilinear: bool,
            features_start: int = 64,
            lr: float = 0.001,
            log_tb_imgs: bool = False,
            tb_img_freq: int = 10000,
            **kwargs
    ):
        super().__init__(num_frames, combine_fn, loss, num_layers, bilinear, features_start, lr, log_tb_imgs, tb_img_freq, ** kwargs)
        self.save_hyperparameters()
        self.num_frames = num_frames
        self.loss = loss

        self.criterion = self._determine_loss_fn()
        self.input_channels = 3 * self.num_frames

        self.net = UNetExtraSkip(
            input_channels = self.input_channels,
            output_channels = 3,
            num_layers = self.hparams.num_layers,
            features_start = self.hparams.features_start,
            bilinear = self.hparams.bilinear)

        self.LPIPS = lpips.LPIPS(net='alex')


if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    print("start right direct model")
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = ExtraSkipModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = RightDaVinciDataModule(
        args.data_dir,
        frames_per_sample=args.num_frames,
        frames_to_drop=0,
        is_color_input=True,
        is_color_output=True,
        extra_info=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()
    print("dm setup")

    # sanity check
    print("size of trainset:", len(dm.train_samples))
    print("size of validset:", len(dm.val_samples))
    print("size of testset:", len(dm.test_samples))

    img, target = next(iter(dm.train_dataloader()))
    print(img.shape)
    print(target.shape)

    # model
    model = ExtraSkipModel(**args.__dict__)
    print("model instance created")
    print("lightning version", pl.__version__)

    # train
    trainer = pl.Trainer.from_argparse_args(args)
    print("trainer created")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
