import os
import random

import pytorch_lightning as pl
from sklearn.utils import shuffle
from torch.utils.data import DataLoader


# target right frames to visualize (hand picked from validation dataset)
VIS_IMG_LIST = [26035, 31777, 19795, 13276, 13356, 13514, 13635, 4742, 31360, 31361, 3337, 21297, 21675, 1364]


class DaVinciDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            frames_per_sample: int,
            frames_to_drop: int = 0,
            is_color_input: bool = True,
            is_color_output: bool = True,
            extra_info: bool = False,
            num_workers: int = 4,
            batch_size: int = 32,
            seed: int = 42,
            num_val_sets: int = 13,
            videos_drop_k: int = None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.frames_per_sample = frames_per_sample
        self.frames_to_drop = frames_to_drop
        self.is_color_input = is_color_input
        self.is_color_output = is_color_output
        self.extra_info = extra_info
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.num_val_sets = num_val_sets
        self.videos_drop_k = videos_drop_k

    # helper
    def _read_image_list(self, filename):
        list_file = open(filename, 'r')
        img_list = []
        while True:
            next_line = list_file.readline()
            if not next_line:
                break
            png_name = next_line.rstrip()
            img_list.append(png_name)
        return img_list[::-1]

    # helper
    def _split_into_chunks(self, img_list, window_size, name):
        all_samples = []
        for i in range(0, len(img_list) - window_size, window_size):
            if i + 2 * window_size > len(img_list):  # if its the last sample
                all_samples.append((name, img_list[i:]))
            else:
                all_samples.append((name, img_list[i: i + window_size]))
        return all_samples

    # helper
    def _sliding_window(self, img_sets, set_name):
        # create samples containing k frames per sample and dropping some number of random frames
        split_samples = []
        step_size = 1  # sample overlap size

        if ((set_name == 'video') and (self.videos_drop_k)):
            end = len(img_sets[0][1]) - self.frames_per_sample + 1 - self.videos_drop_k
        else:
            end = len(img_sets[0][1]) - self.frames_per_sample + 1

        for (name, frame_list) in img_sets:
            cur_set_split_samples = []

            for i in range(0, end, step_size):
                frames = frame_list[i:i + self.frames_per_sample]

                # Randomly drop frames - only do this if we have 3 or more frames
                if self.frames_per_sample > 2:
                    max_frames_to_drop = self.frames_per_sample - 2  # cant drop more than this
                    if self.frames_to_drop > max_frames_to_drop:
                        # TODO: Add warning if user input more frames to drop than makes sense
                        self.frames_to_drop = max_frames_to_drop
                    for i in range(self.frames_to_drop):
                        rand_idx = random.randint(1, len(frames) - 1)
                        _ = frames.pop(rand_idx)

                split_samples += [(name, frames)]

        return split_samples

    # helper
    def _create_val_img_list(self):
        vis_img_samples = []

        # TODO: import note that this only works when we're not randomly dropping frames
        for target_idx in VIS_IMG_LIST:
            idxs = [(target_idx - i) for i in range(self.frames_per_sample)]
            str_idxs = [f"{str(idx).zfill(6)}.png" for idx in idxs]
            vis_img_samples.append(('train', str_idxs))

        return vis_img_samples

    def setup(self):
        # this function does the train/val/test splits - needs to be run first after instantiating dm

        # read files in train folder, reverse order and split into snippets of 500 frames
        train_img_list = self._read_image_list(os.path.join(self.data_dir, 'train.txt'))
        train_sets = self._split_into_chunks(train_img_list, window_size=500, name='train')

        # split snippets into train / val
        train_sets = shuffle(train_sets, random_state=self.seed)
        val_sets = train_sets[:self.num_val_sets]
        train_sets = train_sets[self.num_val_sets:]

        # read files from test folder
        test_img_list = self._read_image_list(os.path.join(self.data_dir, 'test.txt'))
        test_sets = [("test", test_img_list)]

        # apply sliding window to each set
        self.train_samples = self._sliding_window(train_sets, 'train')
        self.val_samples = self._sliding_window(val_sets, 'val')
        self.test_samples = self._sliding_window(test_sets, 'test')
        self.video_samples = self._sliding_window(val_sets, 'video')

        # create separate list of validation images to visualize (hand-picked 14 indices)
        self.vis_samples = self._create_val_img_list()

        # reverse to be from first to last frame for predicting videos since we don't shuffle
        self.val_samples = self.val_samples[::-1]
        self.video_samples = self.video_samples[::-1]

        self.train_dataset = self.dataset(data_dir=self.data_dir,
                                          sample_list=self.train_samples,
                                          frames_per_sample=self.frames_per_sample,
                                          frames_to_drop=self.frames_to_drop,
                                          is_color_input=self.is_color_input,
                                          is_color_output=self.is_color_output,
                                          extra_info=self.extra_info)

        self.val_dataset = self.dataset(data_dir=self.data_dir,
                                        sample_list=self.val_samples,
                                        frames_per_sample=self.frames_per_sample,
                                        frames_to_drop=self.frames_to_drop,
                                        is_color_input=self.is_color_input,
                                        is_color_output=self.is_color_output,
                                        extra_info=self.extra_info)

        self.test_dataset = self.dataset(data_dir=self.data_dir,
                                         sample_list=self.test_samples,
                                         frames_per_sample=self.frames_per_sample,
                                         frames_to_drop=self.frames_to_drop,
                                         is_color_input=self.is_color_input,
                                         is_color_output=self.is_color_output,
                                         extra_info=self.extra_info)

        self.vis_dataset = self.dataset(data_dir=self.data_dir,
                                        sample_list=self.vis_samples,
                                        frames_per_sample=self.frames_per_sample,
                                        frames_to_drop=self.frames_to_drop,
                                        is_color_input=self.is_color_input,
                                        is_color_output=self.is_color_output,
                                        extra_info=self.extra_info)

        self.video_dataset = self.dataset(data_dir=self.data_dir,
                                        sample_list=self.video_samples,
                                        frames_per_sample=self.frames_per_sample,
                                        frames_to_drop=self.frames_to_drop,
                                        is_color_input=self.is_color_input,
                                        is_color_output=self.is_color_output,
                                        extra_info=self.extra_info)

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            pin_memory=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=True)
        return loader

    def val_dataloader_shuffle(self):
        loader = DataLoader(self.val_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            pin_memory=True)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=True)
        return loader

    def vis_img_dataloader(self):
        loader = DataLoader(self.vis_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=True)
        return loader

    def video_dataloader(self):
        loader = DataLoader(self.video_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=True)
        return loader
