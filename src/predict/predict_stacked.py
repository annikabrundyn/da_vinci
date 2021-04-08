import numpy as np
import os
import torch
from tqdm import tqdm
import torchvision

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from models.right_view.stacked_unet2d import StackedModel
from data.multiframe_data import StackedDaVinciDataModule

if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()

    # model args
    parser = StackedModel.add_model_specific_args(parser)
    parser.add_argument("--ckpt", required=True,
                        type=str, help="path to model checkpoint")
    parser.add_argument("--output_dir", required=True,
                        type=str, help="output directory")
    parser.add_argument("--freq", required=True,
                        type=int)

    args = parser.parse_args()

    # model
    model = StackedModel.load_from_checkpoint(checkpoint_path=args.ckpt)
    model.to(device)
    model.eval()
    model.freeze()

    # data
    dm = StackedDaVinciDataModule(
        args.data_dir,
        frames_per_sample=model.hparams.num_frames,
        frames_to_drop=0,
        extra_info=False,
        batch_size=args.batch_size,
        num_workers=model.hparams.num_workers,
    )
    dm.setup()
    print("dm setup")


    outputs = []
    video_idx = 0
    for batch_idx, batch in enumerate(tqdm(dm.val_dataloader())):

        img, target = batch
        img = img.to(device)
        preds = model(img)

        # concat left view and right view
        if model.hparams.num_frames > 1:
            img = img[:, 0:3, ...]

        left_and_right = torch.cat((img, preds), dim=3)

        # following same steps at torchvision save_image
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        left_and_right = torch.mul(left_and_right, 255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1)
        outputs.append(left_and_right)

        # output in chunks to avoid memory errors
        if (batch_idx + 1) % args.freq == 0:
            outputs_tensor = torch.cat(outputs).cpu()
            torchvision.io.write_video(filename=os.path.join(args.output_dir, f"{video_idx}.mp4"),
                                       video_array=outputs_tensor,
                                       video_codec='h264',
                                       fps=20)
            outputs = []
            video_idx += 1

        # last batch
        if (batch_idx + 1) == len(dm.val_dataloader()):
            print("last batch:", batch_idx)
            outputs_tensor = torch.cat(outputs).cpu()
            torchvision.io.write_video(filename=os.path.join(args.output_dir, f"{video_idx}.mp4"),
                                       video_array=outputs_tensor,
                                       video_codec='h264',
                                       fps=20)
            outputs = []

    print("batch:", batch_idx)


