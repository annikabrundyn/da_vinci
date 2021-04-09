import numpy as np
import os
import torch
from tqdm import tqdm
import torchvision

import glob

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from models.right_view.unstacked_unet2d import UnstackedModel
from models.right_view.stacked_unet2d import StackedModel
from data.multiframe_data import UnstackedDaVinciDataModule, StackedDaVinciDataModule

from moviepy.editor import VideoFileClip, concatenate_videoclips


def concat_left_right(img, preds, num_frames, stacked=False):
    if stacked:
        if num_frames > 1:
            img = img[:, 0:3, ...]

        left_and_right = torch.cat((img, preds), dim=3)
    else:
        img = img[:, 0, ...]
        left_and_right = torch.cat((img, preds), dim=3)
    return left_and_right


def ffmpeg_concat_videos():
    stringa = "ffmpeg -i \"concat:"
    elenco_video = glob.glob("*.mp4")
    elenco_file_temp = []
    for f in elenco_video:
        file = "temp" + str(elenco_video.index(f) + 1) + ".ts"
        os.system("ffmpeg -i " + f + " -c copy -bsf:v h264_mp4toannexb -f mpegts " + file)
        elenco_file_temp.append(file)
    print(elenco_file_temp)
    for f in elenco_file_temp:
        stringa += f
        if elenco_file_temp.index(f) != len(elenco_file_temp) - 1:
            stringa += "|"
        else:
            stringa += "\" -c copy  -bsf:a aac_adtstoasc output.mp4"
    print(stringa)
    os.system(stringa)





if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()

    # model args
    parser = UnstackedModel.add_model_specific_args(parser)
    parser.add_argument("--ckpt", required=True, type=str, help="path to model checkpoint")
    parser.add_argument("--output_dir", required=True, type=str, help="output directory")
    parser.add_argument("--freq", required=True, type=int, help="how frequently to save video snippets")
    parser.add_argument("--max_frame_exp", type=int, default=10)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--video_format", type=str, default='mp4')
    parser.add_argument("--stacked", action="store_true")

    args = parser.parse_args()

    # make prediction folder if doesnt exist
    if not os.path.exists('output_dir'):
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
        extra_info=True,
        batch_size=args.batch_size,
        num_workers=model.hparams.num_workers,
        videos_drop_k=(args.max_frame_exp - model.hparams.num_frames),
    )
    dm.setup()
    print("dm setup")

    outputs = []
    video_idx = 0
    for batch_idx, batch in enumerate(tqdm(dm.video_dataloader())):

        if batch_idx > 3:
            break

        img, target, extra_info = batch
        img = img.to(device)
        preds = model(img)

        # concat left view and right view
        left_and_right = concat_left_right(img, preds, stacked=args.stacked, num_frames=model.hparams.num_frames)

        # following same steps at torchvision save_image
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        left_and_right = torch.mul(left_and_right, 255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1)
        outputs.append(left_and_right)

        # output in chunks to avoid memory errors
        if (batch_idx + 1) % args.freq == 0:
            outputs_tensor = torch.cat(outputs).cpu()
            torchvision.io.write_video(filename=os.path.join(args.output_dir, f"{video_idx}.{args.video_format}"),
                                       video_array=outputs_tensor,
                                       video_codec='h264',
                                       fps=args.fps)
            outputs = []
            video_idx += 1

        # last batch
        if (batch_idx + 1) == len(dm.val_dataloader()):
            print("last batch:", batch_idx)
            outputs_tensor = torch.cat(outputs).cpu()
            torchvision.io.write_video(filename=os.path.join(args.output_dir, f"{video_idx}.{args.video_format}"),
                                       video_array=outputs_tensor,
                                       video_codec='h264',
                                       fps=args.fps)
            outputs = []

    print("last batch:", batch_idx)

    print("now concatenate video snippets")
    full_video = []

    for file in os.listdir(args.output_dir):
        if os.path.splitext(file)[1] == args.video_format:
            filePath = os.path.join(args.output_dir, file)
            video = VideoFileClip(filePath)
            full_video.append(video)

    full_video = concatenate_videoclips(full_video)
    full_video.to_videofile("output.mp4", fps=args.fps, remove_temp=False)



