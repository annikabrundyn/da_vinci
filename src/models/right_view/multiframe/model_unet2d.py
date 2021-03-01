from argparse import ArgumentParser

import pytorch_lightning as pl
import lpips

from torchvision.utils import make_grid

from models.right_view.base_model import BaseModel
from models.unet import UNet
from data.right_data import RightDaVinciDataModule
from metrics import FIDCallback


class UNet2DModel(BaseModel):
    def __init__(
            self,
            num_frames: int,
            combine_fn: None,
            loss: str,
            num_layers: int,
            bilinear: bool,
            features_start: int = 64,
            lr: float = 0.001,
            log_tb_imgs: bool = True,
            tb_img_freq: int = 10000,
            **kwargs
    ):
        super().__init__(num_frames, combine_fn, loss, num_layers, bilinear, features_start, lr, log_tb_imgs, tb_img_freq, ** kwargs)

        self.save_hyperparameters()
        self.num_frames = num_frames
        self.loss = loss

        self.criterion = self._determine_loss_fn()

        self.input_channels = 3 * self.num_frames

        self.net = UNet(
            input_channels = self.input_channels,
            output_channels = 3,
            num_layers = self.hparams.num_layers,
            features_start = self.hparams.features_start,
            bilinear = self.hparams.bilinear)

        self.LPIPS = lpips.LPIPS(net='alex')

    def _log_images(self, img, target, pred, step_name):
        # unstack multiple frames to visualize
        c, h, w = img.shape
        img = img.reshape(c // 3, 3, h, w)

        self.logger.experiment.add_image(f'{step_name}_input_left', make_grid(img), self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_target_right', make_grid(target), self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_pred_right', make_grid(pred), self.trainer.global_step)


if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    print("start right multiframe model")
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = UNet2DModel.add_model_specific_args(parser)
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
    model = UNet2DModel(**args.__dict__)
    print("model instance created")
    print("lightning version", pl.__version__)

    # fid metric callback
    #fid = FIDCallback("real_stats.pickle", dm.val_dataloader_shuffle(), args.fid_n_samples, args.fid_epoch_freq)

    # train
    #trainer = pl.Trainer.from_argparse_args(args, callbacks=[fid], log_every_n_steps=1)
    trainer = pl.Trainer.from_argparse_args(args, log_every_n_steps=1)
    print("trainer created")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())