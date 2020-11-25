import numpy as np
import os
import torch
from tqdm import tqdm
import torchvision

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from data.right_data import RightDaVinciDataModule
from models.right_view.direct_right_model import DirectRightModel


if __name__ == "__main__":
    print("Right direct predictions")
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = DirectRightModel.add_model_specific_args(parser)
    parser.add_argument("--ckpt", required=True,
                        type=str, help="path to model checkpoint")
    parser.add_argument("--output_dir", required=True,
                        type=str, help="output directory")
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
    dm.setup_predictions()
    print("dm setup")

    # sanity check
    print("size of prediction samples:", len(dm.val_samples))

    img, target, extra_info = next(iter(dm.val_dataloader()))
    print(img.shape)
    print(target.shape)
    print(f"extra info len: {len(extra_info)}")

    # model
    model = DirectRightModel(**args.__dict__)
    model = DirectRightModel.load_from_checkpoint(checkpoint_path=args.ckpt)
    model.to(device)
    model.eval()
    model.freeze()

    print("model instance created")
    print("lightning version", pl.__version__)

    outputs = []
    for idx, batch in enumerate(tqdm(dm.val_dataloader())):
        img, target, extra_info = batch
        img = img.to(device)
        preds = model(img)

        # concat left view and right view
        left_and_right = torch.cat((img, preds), dim=3).permute(0,2,3,1)
        left_and_right = torch.clamp(left_and_right, 0, 1) * 255
        outputs.append(left_and_right)

        # output in chunks to avoid memory errors
        if (idx + 1) % 20 == 0:
            outputs = torch.cat(outputs)
            torchvision.io.write_video(filename=os.path.join(args.output_dir, f"{idx}.avi"), video_array=outputs, fps=30)
            outputs = []
        outputs = torch.cat(outputs)
        torchvision.io.write_video(filename=os.path.join(args.output_dir, f"{idx}.avi"), video_array=outputs, fps=30)
