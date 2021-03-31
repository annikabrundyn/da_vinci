import numpy as np
import os
import torch
from tqdm import tqdm
import torchvision

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from models.right_view.unstacked_unet2d import UnstackedModel
from data.multiframe_data import UnstackedDaVinciDataModule

if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()

    # model args
    parser = UnstackedModel.add_model_specific_args(parser)
    parser.add_argument("--ckpt", required=True,
                        type=str, help="path to model checkpoint")
    parser.add_argument("--output_dir", required=True,
                        type=str, help="output directory")
    args = parser.parse_args()

    # model
    model = UnstackedModel.load_from_checkpoint(checkpoint_path=args.ckpt)
    model.to(device)
    model.eval()
    model.freeze()

    # data
    dm = UnstackedDaVinciDataModule(
        args.data_dir,
        frames_per_sample=model.hparams.num_frames,
        frames_to_drop=0,
        extra_info=True,
        batch_size=model.hparams.batch_size,
        num_workers=model.hparams.num_workers,
    )
    dm.setup()
    print("dm setup")

    outputs = []
    for idx, batch in enumerate(tqdm(dm.val_dataloader())):
        img, target, extra_info = batch
        img = img.to(device)
        preds = model(img)

        # concat left view and right view
        left_and_right = torch.cat((img[:, 0, ...], preds), dim=3).permute(0,2,3,1)
        left_and_right = torch.clamp(left_and_right, 0, 1) * 255
        outputs.append(left_and_right)

        # output in chunks to avoid memory errors
        if (idx + 1) % 5 == 0:
            outputs = torch.cat(outputs)
            torchvision.io.write_video(filename=os.path.join(args.output_dir, f"{idx}.avi"), video_array=outputs, fps=30)
            outputs = []
        outputs = torch.cat(outputs)
        torchvision.io.write_video(filename=os.path.join(args.output_dir, f"{idx}.avi"), video_array=outputs, fps=30)