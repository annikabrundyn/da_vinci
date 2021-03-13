from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models.right_view.base_model import BaseModel
from models.unet_architecture import UnstackedUNet, UnstackedUNetExtraSkip
from data.multiframe_data import UnstackedDaVinciDataModule
from metrics import FIDCallback
from callbacks import SaveImgCallBack


class UnstackedModel(BaseModel):
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
            print("Normal UNet *without* extra skip connection")
            self.net = UnstackedUNet(num_frames=self.num_frames,
                                     combine_fn=self.combine_fn,
                                     num_layers=self.hparams.num_layers,
                                     features_start=self.hparams.features_start,
                                     bilinear=self.bilinear,
                                     sigmoid_on_output=self.hparams.sigmoid_on_output)

        else:
            print("Modified UNet *with* extra skip connection")
            self.net = UnstackedUNetExtraSkip(num_frames=self.num_frames,
                                              combine_fn=self.combine_fn,
                                              num_layers=self.hparams.num_layers,
                                              features_start=self.hparams.features_start,
                                              bilinear=self.bilinear,
                                              sigmoid_on_output=self.hparams.sigmoid_on_output)

if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    print("start right multiframe model")
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = UnstackedModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = UnstackedDaVinciDataModule(
        args.data_dir,
        frames_per_sample=args.num_frames,
        frames_to_drop=0,
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
    # if args.ckpt_path is not None:
    #     model = UnstackedModel.load_from_checkpoint(args.ckpt_path)
    # else:
    model = UnstackedModel(**args.__dict__)

    print("model instance created")
    print("lightning version", pl.__version__)

    # fid callback
    fid = FIDCallback(pickle_dir=args.data_dir,
                      pickle_name="real_stats.pickle",
                      val_dl=dm.val_dataloader_shuffle(),
                      num_samples=args.fid_n_samples,
                      fid_freq=args.fid_epoch_freq)

    # save val imgs callback
    save_preds = SaveImgCallBack(dm.vis_img_dataloader(), args.save_epoch_freq)

    # 3. Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint = ModelCheckpoint(monitor='val_loss',
                                 filename='{epoch:03d}-{val_loss:.4f}',
                                 save_last=True,
                                 mode="min")

    # train - default logging every 50 steps
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint, fid, save_preds], num_sanity_val_steps=0, gpus=1)
    print("trainer created")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())