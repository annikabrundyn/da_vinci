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