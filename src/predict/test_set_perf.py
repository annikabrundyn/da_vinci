import numpy as np
import os
import torch
from tqdm import tqdm
from argparse import ArgumentParser

import pytorch_lightning as pl
import torchvision

from models.right_view.unstacked_unet2d import UnstackedModel
from models.right_view.stacked_unet2d import StackedModel
from data.multiframe_data import UnstackedDaVinciDataModule, StackedDaVinciDataModule
from pytorch_lightning import Trainer


def test_set_performance(data_dir, ckpt, batch_size=16, unstacked=True, max_frame_exp=10):

    pl.seed_everything(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if unstacked:
        m = UnstackedModel
        d = UnstackedDaVinciDataModule

    # model
    model = m.load_from_checkpoint(checkpoint_path=ckpt)
    model.to(device)
    model.eval()
    model.freeze()

    # data
    dm = d(
        data_dir,
        frames_per_sample=model.hparams.num_frames,
        frames_to_drop=0,
        extra_info=True,
        batch_size=batch_size,
        num_workers=model.hparams.num_workers,
        videos_drop_k=(max_frame_exp - model.hparams.num_frames),
    )
    dm.setup()
    print("dm setup")

    trainer = Trainer.from_argparse_args(*model.hparams)
    results = trainer.test(model=model, test_dataloaders=dm.val_dataloader())
    return results