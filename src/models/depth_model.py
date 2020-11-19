from argparse import ArgumentParser

import pytorch_lightning as pl
from data.depth_data import DepthDaVinciDataModule
from models.model import Model
from models.callbacks.save_pred_img_callback import SavePredImgCallback


class DepthModel(Model):
    def __init__(
            self,
            frames_per_sample: int,
            frames_to_drop: int,
            include_right_view: bool = False,
            stack_horizontal: bool = False,
            is_color_input: bool = False,
            is_color_output: bool = False,
            num_classes: int = 1,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            lr: float = 0.001,
            output_img_freq: int = 500,
            fid_freq: int = 500,
            *args,
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
                         *args,
                         **kwargs)

if __name__ == '__main__':
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    print("start")
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = DepthModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = DepthDaVinciDataModule(args.data_dir,
                             frames_per_sample=args.frames_per_sample,
                             frames_to_drop=args.frames_to_drop,
                             include_right_view=args.include_right_view,
                             stack_horizontal=args.stack_horizontal,
                             is_color_input=args.is_color_input,
                             extra_info=True,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)
    dm.setup()
    print("dm setup")

    # sanity check
    print("size of trainset:", len(dm.train_samples))
    print("size of validset:", len(dm.val_samples))
    print("size of testset:", len(dm.test_samples))

    img, target, extra_info = next(iter(dm.train_dataloader()))
    print(img.shape)
    print(target.shape)
    print(len(extra_info))

    # model
    model = DepthModel(**args.__dict__)
    print("model instance created")
    print('lightning version', pl.__version__)

    # train
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[SavePredImgCallback(dm.vis_img_dataloader())])
    print("trainer created")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    # predict + save val images
    # trainer.test(model, dm.vis_img_dataloader())
