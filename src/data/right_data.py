import math
import os

from sklearn.utils import shuffle
from data.data import DaVinciDataSet, DaVinciDataModule


class RightDaVinciDataSet(DaVinciDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_dir = 'image_1'


class RightDaVinciDataModule(DaVinciDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = RightDaVinciDataSet

    def setup_predictions(self):
        # this function does the train/val/test splits - needs to be run first after instantiating dm
        train_img_list = self._read_image_list(os.path.join(self.data_dir, 'train.txt'))
        train_img_list = train_img_list[::-1]
        self.all_sets = self._split_into_chunks(train_img_list, window_size=1000, name='train')

        test_img_list = self._read_image_list(os.path.join(self.data_dir, 'test.txt'))
        test_img_list = test_img_list[::-1]
        self.all_sets += self._split_into_chunks(test_img_list, window_size=1000, name='test')

        # split train/val/test
        val_len = math.floor(self.val_split * len(self.all_sets))
        test_len = math.floor(self.test_split * len(self.all_sets))
        train_len = len(self.all_sets) - val_len - test_len

        self.train_sets = self.all_sets[:train_len]
        self.val_sets = self.all_sets[train_len:train_len + val_len]
        self.test_sets = self.all_sets[-test_len:]

        # create separate list of random images from validation set to predict on at the end of training
        self.vis_img_list_names = self._create_val_img_list(self.val_sets)
        self.vis_img_list = []

        self.train_samples = self._sliding_window(self.train_sets)
        self.val_samples = self._sliding_window(self.val_sets, val_set=True)
        self.test_samples = self._sliding_window(self.test_sets)

        self.train_dataset = self.dataset(data_dir=self.data_dir,
                                          sample_list=self.train_samples,
                                          frames_per_sample=self.frames_per_sample,
                                          frames_to_drop=self.frames_to_drop,
                                          include_right_view=self.include_right_view,
                                          stack_horizontal=self.stack_horizontal,
                                          is_color_input=self.is_color_input,
                                          is_color_output=self.is_color_output,
                                          extra_info=self.extra_info)

        self.val_dataset = self.dataset(data_dir=self.data_dir,
                                        sample_list=self.val_samples,
                                        frames_per_sample=self.frames_per_sample,
                                        frames_to_drop=self.frames_to_drop,
                                        include_right_view=self.include_right_view,
                                        stack_horizontal=self.stack_horizontal,
                                        is_color_input=self.is_color_input,
                                        is_color_output=self.is_color_output,
                                        extra_info=self.extra_info)

        self.test_dataset = self.dataset(data_dir=self.data_dir,
                                         sample_list=self.test_samples,
                                         frames_per_sample=self.frames_per_sample,
                                         frames_to_drop=self.frames_to_drop,
                                         include_right_view=self.include_right_view,
                                         stack_horizontal=self.stack_horizontal,
                                         is_color_input=self.is_color_input,
                                         is_color_output=self.is_color_output,
                                         extra_info=self.extra_info)

        self.vis_dataset = self.dataset(data_dir=self.data_dir,
                                        sample_list=self.vis_img_list,
                                        frames_per_sample=self.frames_per_sample,
                                        frames_to_drop=self.frames_to_drop,
                                        include_right_view=self.include_right_view,
                                        stack_horizontal=self.stack_horizontal,
                                        is_color_input=self.is_color_input,
                                        is_color_output=self.is_color_output,
                                        extra_info=self.extra_info)
