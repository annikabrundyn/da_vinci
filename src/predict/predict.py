import os
from argparse import ArgumentParser

from tqdm import tqdm

import torch
import pytorch_lightning as pl
from torchvision.utils import save_image
import torchvision.io

from models.right_view.unstacked_unet2d import UnstackedModel
from data.multiframe_data import UnstackedDaVinciDataModule


if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='.', help="path to davinci data")
    parser.add_argument("--ckpt_path", type=str, help="path to saved model checkpoint")
    parser.add_argument("--save_path", type=str, help="path to saved images")
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    model = UnstackedModel.load_from_checkpoint(args.ckpt_path)
    model.eval()

    dm = UnstackedDaVinciDataModule(
        args.data_dir,
        frames_per_sample=model.hparams.num_frames,
        frames_to_drop=0,
        extra_info=True,
        batch_size=1,
        num_workers=model.hparams.num_frames,
    )
    dm.setup()

    #outputs = []
    for idx, (inputs, _, target_frame_name) in enumerate(tqdm(dm.val_dataloader())):
        pred = model(inputs)
        left_and_right = torch.cat((inputs[:, 0, ...], pred), dim=3)
        #outputs.append(left_and_right)
        pred_path = os.path.join(args.save_path, target_frame_name[0])
        save_image(left_and_right, fp=pred_path)

        # output in chunks to avoid memory errors
        # if (idx) % 2 == 0:
        #     outputs = torch.cat(outputs)
        #     torchvision.io.write_video(filename=os.path.join(args.save_path, f"{idx}.avi"),
        #                                video_array=outputs,
        #                                fps=30)
        #     outputs = []

        #outputs = torch.cat(outputs)
        #torchvision.io.write_video(filename=os.path.join(args.save_path, f"{idx}.avi"), video_array=outputs, fps=30)

print("hi")
