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
                 frames_per_sample: int =3,
                 resize: float = 1,
                 img_transform = None,
                 target_transform = None
                 ):
        self.root_dir = root_dir
        self.image_set = image_set

        #new_height = round(480*resize)
        #new_width = round(640*resize)

        if not img_transform:
            self.img_transform = transforms.Compose([transforms.Grayscale(),
                                                     #transforms.Resize((new_height, new_width)),
                                                     transforms.ToTensor()])
        else:
            self.img_transform = img_transform

        if not target_transform:
            self.target_transform = transforms.Compose([#transforms.Resize((new_height, new_width)),
                                                        transforms.ToTensor()])
        else:
            self.target_transform = target_transform

        # create dict with each video name (of diff. scenes) as a key and a list of corresponding frames for that video
        self.frames_per_sample = frames_per_sample

        # img list is already sorted in the correct order
        # TODO: add a check for this
        img_list = self.read_image_list(os.path.join(root_dir, '{:s}.txt'.format(image_set)))

        # create samples containing k frames per sample
        self.all_samples = []
        if self.frames_per_sample > 1:
            step_size = 1 # sample overlap size
            for i in range(0, len(img_list)-self.frames_per_sample, step_size):
                frames = img_list[i:i+self.frames_per_sample]
                # Randomly drop one frame to reduce correlation between samples
                if self.frames_per_sample > 2:
                    rand_idx = random.randint(0, self.frames_per_sample - 2)
                    _ = frames.pop(rand_idx)
                self.all_samples += [frames]

        # only using single frame
        else:
            self.all_samples += [[i] for i in img_list]

        print("len of all samples:", len(self.all_samples))

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
        for frame in frames:
            img_path = os.path.join(self.root_dir, '{:s}'.format(self.image_set), 'image_0', '{:s}'.format(frame))
            image = Image.open(img_path)
            image = self.img_transform(image)
            images.append(image)

        # channels are the b/w input frames
        image_tensor = torch.stack(images)
        image_tensor = torch.squeeze(image_tensor, 1)

        # target is only the last frame
        target_path = os.path.join(self.root_dir, '{:s}'.format(self.image_set), 'disparity', '{:s}'.format(frames[-1]))
        target = Image.open(target_path)
        target = self.target_transform(target)

        return image_tensor, target


class DaVinciDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            frames_per_sample: int = 1,
            resize: float = 1,
            val_split: float = 0.2,
            num_workers: int = 4,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.frames_per_sample = frames_per_sample
        self.resize = resize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset = DaVinciDataSet(self.data_dir,
                                            frames_per_sample=self.frames_per_sample,
                                            resize=self.resize,
                                            image_set='train')

        self.val_dataset = DaVinciDataSet(self.data_dir,
                                          frames_per_sample=self.frames_per_sample,
                                          resize=self.resize,
                                          image_set='test')

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

    # def test_dataloader(self):
    #     loader = DataLoader(self.testset,
    #                         batch_size=self.batch_size,
    #                         shuffle=False,
    #                         num_workers=self.num_workers)
    #     return loader

ds = DaVinciDataSet('/Users/annikabrundyn/Developer/da_vinci_depth/daVinci_data')
dm = DaVinciDataModule('/Users/annikabrundyn/Developer/da_vinci_depth/daVinci_data')
print("hi")