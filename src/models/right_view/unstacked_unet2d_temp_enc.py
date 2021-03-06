from argparse import ArgumentParser

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models.right_view.base_model import BaseModel
from models.unet_architecture import UnstackedTempEncUNet
from data.multiframe_data import UnstackedDaVinciDataModule
from metrics import FIDCallback
from callbacks import SaveImgCallBack


class UnstackedTempEncModel(BaseModel):
    def __init__(
            self,
            num_frames: int,
            combine_fn: str,
            loss: str,
            extra_skip: str,
            num_layers: int,
            bilinear: str,
            sigmoid_on_output: bool,
            features_start: int = 64,
            lr: float = 0.001,
            log_tb_imgs: bool = True,
            tb_img_freq: int = 10000,
            checkpoint_dir: str = None,
            **kwargs
    ):
        super().__init__(num_frames, combine_fn, loss, extra_skip, num_layers, bilinear, sigmoid_on_output,
                         features_start, lr, log_tb_imgs, tb_img_freq, ** kwargs)

        # UNet without extra skip connection (normal)
        if self.hparams.extra_skip in ("False", "F", "false"):
            print("Architecture: Normal UNet *without* extra skip connection")
            self.net = UnstackedTempEncUNet(num_frames=self.num_frames,
                                            combine_fn=self.combine_fn,
                                            num_layers=self.hparams.num_layers,
                                            features_start=self.hparams.features_start,
                                            bilinear=self.bilinear,
                                            sigmoid_on_output=self.hparams.sigmoid_on_output)

        else:
            print("not possible to add extra skip")
            # print("Architecture: Modified UNet *with* extra skip connection")
            self.net = UnstackedTempEncUNet(num_frames=self.num_frames,
                                            combine_fn=self.combine_fn,
                                            num_layers=self.hparams.num_layers,
                                            features_start=self.hparams.features_start,
                                            bilinear=self.bilinear,
                                            sigmoid_on_output=self.hparams.sigmoid_on_output)

        self.save_hyperparameters()


if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = UnstackedTempEncModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # initialize model, load from checkpoint if passed and update saved dm parameters
    if args.ckpt_path is None:
        print("no model checkpoint provided")
        model = UnstackedTempEncModel(**args.__dict__)
    else:
        print("load pretrained model checkpoint")
        # only parameter that we change is the learning rate provided
        model = UnstackedTempEncModel.load_from_checkpoint(args.ckpt_path, lr=args.lr)
        args.data_dir = model.hparams.data_dir
        args.num_frames = model.hparams.num_frames
        args.batch_size = model.hparams.batch_size
        args.num_workers = model.hparams.num_workers

    print("initialize datamodule...")
    dm = UnstackedDaVinciDataModule(
        args.data_dir,
        frames_per_sample=args.num_frames,
        frames_to_drop=0,
        extra_info=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    # sanity check
    print("size of trainset:", len(dm.train_samples))
    print("size of validset:", len(dm.val_samples))
    print("size of testset:", len(dm.test_samples))

    img, target = next(iter(dm.train_dataloader()))
    print('input shape: ', img.shape)
    print('target shape: ', target.shape)

    # fid callback
    fid = FIDCallback(pickle_dir=args.data_dir,
                      pickle_name="real_stats.pickle",
                      val_dl=dm.val_dataloader(),
                      num_samples=args.fid_n_samples,
                      fid_freq=args.fid_epoch_freq)

    # save val imgs callback
    save_preds = SaveImgCallBack(dm.vis_img_dataloader(), args.save_epoch_freq)

    # model checkpoint callback
    checkpoint = ModelCheckpoint(monitor='val_loss',
                                 filename='{epoch:03d}-{val_loss:.4f}',
                                 save_last=True,
                                 mode="min")

    # init pl trainer
    print("initialize trainer")
    # note removed resume from checkpoint in trainer
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint, fid, save_preds], num_sanity_val_steps=0)

    print("start training model...")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())