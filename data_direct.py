from data_v2 import DaVinciDataSet, DaVinciDataModule

import os
import random
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split

import pytorch_lightning as pl
from PIL import Image


class DirectDaVinciDataSet(DaVinciDataSet):
    def __init__(
        self,
        data_dir: str,
        sample_list: list,
        frames_per_sample: int,
        frames_to_drop: int,
        img_transform=None,
        target_transform=None,
    ):

        super().__init__(data_dir,
        sample_list,
        frames_per_sample,
        frames_to_drop,
        False,  # whether to include right images
        img_transform,
        target_transform)

        if not img_transform:
            self.img_transform = transforms.Compose(
                [transforms.ToTensor()]
            )

    def __getitem__(self, index):
        image_set, frames = self.sample_list[index]

        images = []

        for frame in frames:
            img_path = os.path.join(self.data_dir, image_set, "image_0", frame)
            image = Image.open(img_path)
            image = self.img_transform(image)
            images.append(image)

        # TODO: NO LONGER TRUE - channels are the b/w input frames
        image_tensor = torch.cat(images)

        # target is only the first frame
        target_path = os.path.join(self.data_dir, image_set, "image_1", frame)
        target = Image.open(target_path)
        target = self.target_transform(target)

        return image_tensor, target


class DirectDaVinciDataModule(DaVinciDataModule):
    def __init__(
        self,
        data_dir: str,
        frames_per_sample: int,
        frames_to_drop: int,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 4,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(data_dir,
        frames_per_sample,
        frames_to_drop,
        False,
        val_split,
        test_split,
        num_workers,
        batch_size, *args, **kwargs)


    def setup(self):

        train_img_list = self._read_image_list(os.path.join(self.data_dir, "train.txt"))
        train_img_list = train_img_list[::-1]
        all_sets = self._split_into_chunks(
            train_img_list, window_size=1000, name="train"
        )

        test_img_list = self._read_image_list(os.path.join(self.data_dir, "test.txt"))
        test_img_list = test_img_list[::-1]
        all_sets += self._split_into_chunks(
            test_img_list, window_size=1000, name="test"
        )

        # shuffle all 41 sets of 1000 frames
        random.shuffle(all_sets)

        # split train/val/test
        val_len = math.floor(self.val_split * len(all_sets))
        test_len = math.floor(self.test_split * len(all_sets))
        train_len = len(all_sets) - val_len - test_len

        train_sets = all_sets[:train_len]
        val_sets = all_sets[train_len : train_len + val_len]
        test_sets = all_sets[-test_len:]

        self.train_samples = self._sliding_window(train_sets)
        self.val_samples = self._sliding_window(val_sets)
        self.test_samples = self._sliding_window(test_sets)

        random.shuffle(self.train_samples)
        random.shuffle(self.val_samples)
        random.shuffle(self.test_samples)

        self.train_dataset = DirectDaVinciDataSet(
            data_dir=self.data_dir,
            sample_list=self.train_samples,
            frames_per_sample=self.frames_per_sample,
            frames_to_drop=self.frames_to_drop,
        )

        self.val_dataset = DirectDaVinciDataSet(
            data_dir=self.data_dir,
            sample_list=self.val_samples,
            frames_per_sample=self.frames_per_sample,
            frames_to_drop=self.frames_to_drop,
        )

        self.test_dataset = DirectDaVinciDataSet(
            data_dir=self.data_dir,
            sample_list=self.test_samples,
            frames_per_sample=self.frames_per_sample,
            frames_to_drop=self.frames_to_drop,
        )
