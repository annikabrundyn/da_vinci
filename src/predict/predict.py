"""
[WIP] Script to predict given a specific model, checkpoint path and dataset
TODO:
    - add cli arguments
    - test on all our models + dataloader
    - generalize to take any model
"""
import yaml

import pytorch_lightning as pl
from models.right_view.stacked_unet2d import StackedModel
from data import StackedDaVinciDataModule

CKPT_PATH = "/Users/annikabrundyn/Downloads/epoch=2-step=5453.ckpt"
HPARAMS_PATH = "/Users/annikabrundyn/Downloads/hparams.yaml"


with open(HPARAMS_PATH, "r") as stream:
    args = yaml.load(stream)

args['data_dir'] = "/Users/annikabrundyn/Developer/da_vinci/daVinci_data"


dm = StackedDaVinciDataModule(
    args['data_dir'],
    frames_per_sample=args['num_frames'],
    frames_to_drop=0,
    extra_info=False,
    batch_size=args['batch_size'],
    num_workers=args['num_workers'],
)
dm.setup()


model = StackedModel.load_from_checkpoint(
    checkpoint_path=CKPT_PATH,
    hparams_file=HPARAMS_PATH,
    map_location=None
)

trainer = pl.Trainer()

test_results = trainer.test(model, test_dataloaders=dm.vis_img_dataloader())

print(test_results)