import math
import os

from sklearn.utils import shuffle
from data import DaVinciDataSet, DaVinciDataModule


class RightDaVinciDataSet(DaVinciDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_dir = 'image_1'
        # self._img_transforms()


class RightDaVinciDataModule(DaVinciDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = RightDaVinciDataSet




dm = RightDaVinciDataModule(data_dir = "/Users/annikabrundyn/Developer/da_vinci/daVinci_data",
                            frames_per_sample = 1,
                            frames_to_drop = 0,
                            is_color_input=True,
                            is_color_output=True,
                            extra_info=True)
dm.setup()
print('hey')