import os
import random
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split

import pytorch_lightning as pl
from PIL import Image


class DaVinciDataSet(Dataset):

    def __init__(self,
                 root_dir: str,
                 image_set: str = 'train',
                 frames_per_sample: int = 5,
                 frames_to_drop: int = 2,
                 include_right_view: bool = False, #whether to include right images
                 img_transform = None,
                 target_transform = None
                 ):
        self.root_dir = root_dir
        self.image_set = image_set
        self.include_right_view = include_right_view

        if not img_transform:
            self.img_transform = transforms.Compose([transforms.Grayscale(),
                                                     transforms.ToTensor()])
        else:
            self.img_transform = img_transform

        if not target_transform:
            self.target_transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.target_transform = target_transform

        self.frames_per_sample = frames_per_sample
        self.frames_to_drop = frames_to_drop

        # img list is already sorted in the correct order - just a list of the png names
        # TODO: add a check for this
        img_list = self.read_image_list(os.path.join(root_dir, '{:s}.txt'.format(image_set)))
        img_list = img_list[::-1]

        # create samples containing k frames per sample and dropping some number of random frames
        self.all_samples = []
        if self.frames_per_sample > 1:
            step_size = 1 # sample overlap size
            for i in range(0, len(img_list)-self.frames_per_sample+1, step_size):
                frames = img_list[i:i+self.frames_per_sample]

                # Randomly drop frames - only do this if we have 3 or more frames
                if self.frames_per_sample > 2:
                    max_frames_to_drop = self.frames_per_sample - 2  # cant drop more than this
                    if self.frames_to_drop > max_frames_to_drop:
                        #TODO: Add warning if user input more frames to drop than makes sense
                        self.frames_to_drop = max_frames_to_drop
                    for i in range(self.frames_to_drop):
                        rand_idx = random.randint(0, len(frames) - 2)
                        _ = frames.pop(rand_idx)
                self.all_samples += [frames]

        # only using single frame
        else:
            self.all_samples += [[i] for i in img_list]

        # reverse order so predicted frames align
        # self.all_samples = self.all_samples[::-1]

        # shuffle
        # random.shuffle(self.all_samples)

    def read_image_list(self, filename):
        list_file = open(filename, 'r')
        img_list = []
        while True:
            next_line = list_file.readline()
            if not next_line:
                break
            png_name = next_line.rstrip()

            img_list.append(png_name)
        return img_list

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        frames = self.all_samples[index]

        images = []
        if self.include_right_view:
            right_images = []

        for frame in frames:
            img_path = os.path.join(self.root_dir, '{:s}'.format(self.image_set), 'image_0', '{:s}'.format(frame))
            image = Image.open(img_path)
            image = self.img_transform(image)
            images.append(image)

            if self.include_right_view:
                img_path = os.path.join(self.root_dir, '{:s}'.format(self.image_set), 'image_1', '{:s}'.format(frame))
                image = Image.open(img_path)
                image = self.img_transform(image)
                right_images.append(image)

        # channels are the b/w input frames
        image_tensor = torch.cat(images)

        if self.include_right_view:
            right_image_tensor = torch.cat(right_images)
            image_tensor = torch.cat((image_tensor, right_image_tensor), dim=0)

        # target is only the first frame
        target_path = os.path.join(self.root_dir, '{:s}'.format(self.image_set), 'disparity', '{:s}'.format(frames[0]))
        target = Image.open(target_path)
        target = self.target_transform(target)

        return image_tensor, target


class DaVinciDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            frames_per_sample: int = 5,
            frames_to_drop: int = 2,
            include_right_view: bool = False,
            val_split: float = 0.2,
            test_split: float = 0.1,
            num_workers: int = 4,
            batch_size: int = 32,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.frames_per_sample = frames_per_sample
        self.frames_to_drop = frames_to_drop
        self.include_right_view = include_right_view
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.full_train_dataset = DaVinciDataSet(self.data_dir,
                                                 frames_per_sample=self.frames_per_sample,
                                                 frames_to_drop=self.frames_to_drop,
                                                 include_right_view=self.include_right_view,
                                                 image_set='train')

        # take last 3500 samples in train as validation set
        #self.val_dataset = torch.utils.data.Subset(self.full_train_dataset, list(range(val_samples_train)))
        #self.train_dataset = torch.utils.data.Subset(self.)
        #self.train_dataset = self.full_train_dataset[val_samples_train+self.frames_per_sample:]

        self.full_test_dataset = DaVinciDataSet(self.data_dir,
                                                frames_per_sample=self.frames_per_sample,
                                                frames_to_drop=self.frames_to_drop,
                                                include_right_view=self.include_right_view,
                                                image_set='test')

        # take first 3500 samples in test as validation set
        #self.val_dataset = self.full_train_dataset[:val_samples_train]
        #self.train_dataset = self.full_train_dataset[val_samples_train+self.frames_per_sample:]

        # split into train/validation
        #val_len = int(val_split * len(self.train_val_dataset))
        #train_len = len(self.train_val_dataset) - val_len
        #self.train_dataset, self.val_dataset = random_split(self.train_val_dataset, lengths=[train_len, val_len])

    def _read_image_list(self, filename):
        list_file = open(filename, 'r')
        img_list = []
        while True:
            next_line = list_file.readline()
            if not next_line:
                break
            png_name = next_line.rstrip()

            img_list.append(png_name)
        return img_list

    def _split_into_chunks(self, img_list, window_size, name):
        all_samples = []
        for i in range(0, len(img_list) - window_size, window_size):
            if i + 2*window_size > len(img_list):  #if its the last sample
                all_samples.append((name, img_list[i:]))
            else:
                all_samples.append((name, img_list[i: i+window_size]))
        return all_samples



    def setup(self, stage = None):
        train_img_list = self._read_image_list(os.path.join(self.data_dir, 'train.txt'))
        train_img_list = train_img_list[::-1]
        all_samples = self._split_into_chunks(train_img_list, window_size=1000, name='train')

        test_img_list = self._read_image_list(os.path.join(self.data_dir, 'test.txt'))
        test_img_list = test_img_list[::-1]
        all_samples += self._split_into_chunks(test_img_list, window_size=1000, name='test')

        # shuffle all 41 sets of 1000 frames
        random.shuffle(all_samples)

        # split train/val/test
        val_len = math.floor(self.val_split * len(all_samples))
        test_len = math.floor(self.test_split * len(all_samples))
        train_len = len(all_samples) - val_len - test_len

        train_sets = all_samples[:train_len]
        val_sets = all_samples[train_len:train_len+val_len]
        test_sets = all_samples[-test_len:]




        train_sets = all_samples

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader

#ds = DaVinciDataSet('/Users/annikabrundyn/Developer/da_vinci_depth/daVinci_data', frames_per_sample=3, frames_to_drop=0)

dm = DaVinciDataModule('/Users/annikabrundyn/Developer/da_vinci_depth/daVinci_data')
dm.setup()
