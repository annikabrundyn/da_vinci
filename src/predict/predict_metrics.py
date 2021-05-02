import os
import torch
from tqdm import tqdm
from argparse import ArgumentParser

import pytorch_lightning as pl
from torchvision.utils import save_image

from models.right_view.unstacked_unet2d import UnstackedModel
from models.right_view.stacked_unet2d import StackedModel
from data.multiframe_data import UnstackedDaVinciDataModule, StackedDaVinciDataModule



if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = UnstackedModel.add_model_specific_args(parser)
    parser.add_argument("--ckpt", required=True, type=str, help="path to model checkpoint")
    parser.add_argument("--output_dir", required=True, type=str, help="output directory")
    parser.add_argument("--max_frame_exp", type=int, default=10)
    parser.add_argument("--stacked", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    # make prediction folder if doesnt exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.stacked:
        m = StackedModel
        d = StackedDaVinciDataModule
    else:
        m = UnstackedModel
        d = UnstackedDaVinciDataModule

    # model
    model = m.load_from_checkpoint(checkpoint_path=args.ckpt)
    model.to(device)
    model.eval()
    model.freeze()

    # data
    dm = d(
        args.data_dir,
        frames_per_sample=model.hparams.num_frames,
        frames_to_drop=0,
        extra_info=False,
        batch_size=args.batch_size,
        num_workers=model.hparams.num_workers,
        #videos_drop_k=(args.max_frame_exp - model.hparams.num_frames),
    )
    dm.setup()
    print("dm setup")

    if args.test:
        dl = dm.test_dataloader()
    else:
        dl = dm.val_dataloader()

    trainer = pl.Trainer.from_argparse_args(args)
    output = trainer.test(model=model, test_dataloaders=dm.val_dataloader(), verbose=True)

    print(output)