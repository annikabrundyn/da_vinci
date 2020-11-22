from data.data import DaVinciDataModule
from models.depth_map.model import Model

import os.path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from mpl_toolkits.axes_grid1 import ImageGrid
from models.callbacks.save_pred_img_callback import SavePredImgCallback
from models.callbacks.fid_callback import FidCallback


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
                         **kwargs)

    def _matplotlib_imshow_input_imgs(self, img, folder_name, frame_nums, save_fig=False, title=None, trainer=None):
        """Summary

        Args:
            img (tensor): (num_channels, H, W)

        Returns:
            TYPE: fig
        """
        if self.stack_horizontal:
            nrow = self.input_channels
            ncol = 1
        elif not self.stack_horizontal:
            if self.include_right_view:
                nrow = self.input_channels // 2
                ncol = 2
            else:
                nrow = self.input_channels
                ncol = 1

        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(nrow, ncol),
                         direction='column',
                         axes_pad=0.3)

        if self.is_color_input:
            range_inputs = range(self.input_channels // 3)
        else:
            range_inputs = range(self.input_channels)

        for ax, idx in zip(grid, range_inputs):
            if self.is_color_input:
                # select 3 channels for color inputs
                npimg = img[3*idx:3*(idx+1)].squeeze().detach().cpu().numpy()
                npimg = np.transpose(npimg, (1, 2, 0))
                ax.imshow(npimg)
            else:
                npimg = img[idx].squeeze().detach().cpu().numpy()
                ax.imshow(npimg, cmap='gray')
            ax.axis('off')
            side = 'left'
            if idx >= nrow:
                side = 'right'
                idx = idx - nrow
            ax.set_title(f"{side} view: {folder_name}/{frame_nums[idx // 2]}/include_right_view:{self.include_right_view}")

        if save_fig:
            dir = trainer.checkpoint_callback.dirpath
            dir = os.path.split(dir)[0]
            dir_path = os.path.join(dir, f"epoch_{self.current_epoch}", "input")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            img_path = os.path.join(dir_path, f"{title}.png")
            plt.savefig(img_path, bbox_inches='tight')
            plt.close()

        return fig


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
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[SavePredImgCallback(dm.vis_img_dataloader()),FidCallback(dm.train_dataloader(),dm.val_dataloader())])
    print("trainer created")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
