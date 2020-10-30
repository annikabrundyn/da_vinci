import os
import random

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

        # create samples containing k frames per sample and dropping some number of random frames
        self.all_samples = []
        if self.frames_per_sample > 1:
            step_size = 1 # sample overlap size
            for i in range(0, len(img_list)-self.frames_per_sample, step_size):
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

        # shuffle
        random.shuffle(self.all_samples)

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

        # target is only the last frame
        target_path = os.path.join(self.root_dir, '{:s}'.format(self.image_set), 'disparity', '{:s}'.format(frames[-1]))
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
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_val_dataset = DaVinciDataSet(self.data_dir,
                                                frames_per_sample=self.frames_per_sample,
                                                frames_to_drop=self.frames_to_drop,
                                                include_right_view=self.include_right_view,
                                                image_set='train')

        self.test_dataset = DaVinciDataSet(self.data_dir,
                                           frames_per_sample=self.frames_per_sample,
                                           frames_to_drop=self.frames_to_drop,
                                           include_right_view=self.include_right_view,
                                           image_set='test')

        # split into train/validation
        val_len = int(val_split * len(self.train_val_dataset))
        train_len = len(self.train_val_dataset) - val_len

        self.train_dataset, self.val_dataset = random_split(self.train_val_dataset,
                                                            lengths=[train_len, val_len])

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

# ds = DaVinciDataSet('/Users/annikabrundyn/Developer/da_vinci_depth/daVinci_data',
#                     include_right_view=True)
# ds.__getitem__(0)
# dm = DaVinciDataModule('/Users/annikabrundyn/Developer/da_vinci_depth/daVinci_data',
#                        include_right_view=True)
# print("hi")