import math
import os

from sklearn.utils import shuffle
from data.data import DaVinciDataSet, DaVinciDataModule

class RightDaVinciDataSet(DaVinciDataSet):
    def __init__(self,
                 data_dir: str,
                 sample_list: list,
                 frames_per_sample: int,
                 frames_to_drop: int,
                 include_right_view: bool = False,  # whether to include right images
                 stack_horizontal: bool = False,
                 is_color_input: bool = True,
                 extra_info: bool = False,
                 img_transform=None,
                 target_transform=None,
                 target_dir='image_1'
                 ):
        super().__init__(data_dir,
                         sample_list,
                         frames_per_sample,
                         frames_to_drop,
                         include_right_view,  # whether to include right images
                         stack_horizontal,
                         is_color_input,
                         extra_info,
                         img_transform,
                         target_transform,
                         target_dir)


class RightDaVinciDataModule(DaVinciDataModule):
    def __init__(
            self,
            data_dir: str,
            frames_per_sample: int,
            frames_to_drop: int,
            include_right_view: bool = False,
            stack_horizontal: bool = False,
            is_color_input: bool = True,
            extra_info: bool = False,
            val_split: float = 0.2,
            test_split: float = 0.1,
            num_workers: int = 8,
            batch_size: int = 32,
            num_pred_img_samples: int = 15,
            *args,
            **kwargs,
    ):
        super().__init__(data_dir,
                         frames_per_sample,
                         frames_to_drop,
                         include_right_view,
                         stack_horizontal,
                         is_color_input,
                         extra_info,
                         val_split,
                         test_split,
                         num_workers,
                         batch_size,
                         num_pred_img_samples, *args, **kwargs)
        self.dataset = RightDaVinciDataSet
