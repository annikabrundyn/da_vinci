import os
import numpy as np
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from data.right_data import RightDaVinciDataModule
from models.depth_map.model import Model
from models.callbacks.save_pred_img_callback import SavePredImgCallback
from models.callbacks.fid_callback import FidCallback
from models.right_view.right_unet import RightUNet


class RightModel(Model):
    def __init__(
        self,
        frames_per_sample: int = 1,
        frames_to_drop: int = 0,
        include_right_view: bool = False,
        stack_horizontal: bool = False,
        is_color_input: bool = True,
        is_color_output: bool = True,
        num_classes: int = 3,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
        lr: float = 0.001,
        output_img_freq: int = 500,
        **kwargs
    ):
        super().__init__(
            frames_per_sample,
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
            **kwargs
        )

        self.net = RightUNet(num_classes=self.hparams.num_classes,
                             input_channels=self.input_channels,
                             num_stack_horizontal=self.num_stack_horizontal,
                             num_layers=self.hparams.num_layers,
                             features_start=self.hparams.features_start,
                             bilinear=self.hparams.bilinear)

    def _matplotlib_imshow_dm(self, img, title, inverse=False, cmap='magma', save_fig=False, dir_path=None):
        if inverse:
            img = 1 - img
        npimg = img.squeeze().detach().cpu().numpy()
        fig = plt.figure()
        npimg = np.transpose(npimg, (1, 2, 0))
        plt.imshow(npimg)
        plt.title(title)

        if save_fig:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            img_path = os.path.join(dir_path, f"{title}.png")
            plt.savefig(img_path, bbox_inches='tight')
            plt.close()

        return fig

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, help="path to davinci data")
        parser.add_argument(
            "--frames_per_sample",
            type=int,
            default=1,
            help="number of frames to include in each sample",
        )
        parser.add_argument(
            "--frames_to_drop",
            type=int,
            default=0,
            help="number of frames to randomly drop in each sample",
        )
        parser.add_argument(
            "--include_right_view",
            action="store_true",
            default=False,
            help="include left and right view",
        )
        parser.add_argument(
            "--stack_horizontal",
            action="store_true",
            default=False,
            help="stacks input views horizontally",
        )
        parser.add_argument(
            "--is_color_input",
            action="store_true",
            default=True,
            help="use color inputs instead of bw",
        )
        parser.add_argument(
            "--num_classes", type=int, default=3, help="output channels"
        )
        parser.add_argument(
            "--batch_size", type=int, default=16, help="size of the batches"
        )
        parser.add_argument("--output_img_freq", type=int, default=100)
        parser.add_argument("--num_workers", type=int, default=8)
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
    print("start right direct model")
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = RightModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = RightDaVinciDataModule(
        args.data_dir,
        frames_per_sample=args.frames_per_sample,
        frames_to_drop=args.frames_to_drop,
        include_right_view=args.include_right_view,
        stack_horizontal=args.stack_horizontal,
        is_color_input=args.is_color_input,
        extra_info=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
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
    model = RightModel(**args.__dict__)
    print("model instance created")
    print("lightning version", pl.__version__)

    # train
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[SavePredImgCallback(dm.vis_img_dataloader()),FidCallback(dm.train_dataloader(),dm.val_dataloader())])
    print("trainer created")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
