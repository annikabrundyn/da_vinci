import math
import os
import random

import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data.data import DaVinciDataSet, DaVinciDataModule


class StackedDaVinciDataSet(DaVinciDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channel_stack_frames = True


class StackedDaVinciDataModule(DaVinciDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = StackedDaVinciDataSet


class UnstackedDaVinciDataSet(DaVinciDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channel_stack_frames = False


class UnstackedDaVinciDataModule(DaVinciDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = UnstackedDaVinciDataSet
