from src.models.model import DepthMap
from src.data.data import DaVinciDataModule

import pytorch_lightning as pl

pl.seed_everything(42)

model = DepthMap.load_from_checkpoint(frames_per_sample=1, frames_to_drop=0, include_right_view=True, batch_size=32, checkpoint_path='https://gridai-8b9b66dc-38ae-4fe4-9e96-bb78af2e1546.s3.amazonaws.com/grid_artifacts/projects/da_vinci_depth/experiments/test-train-lr-sing-img-v3-exp1/version_0/checkpoints/epoch%3D105.ckpt?AWSAccessKeyId=AKIAUMW2YRMZ4VK7ETMX&Signature=v%2FG2%2BiDgtpPqx3HtmgsGljYq6%2B4%3D&Expires=1604594249')

dm = DaVinciDataModule(data_dir='/Users/annikabrundyn/Developer/da_vinci_depth/daVinci_data',
                       frames_per_sample=1,
                       frames_to_drop=0,
                       include_right_view=True,
                       extra_info=True,
                       batch_size=32)
dm.setup()