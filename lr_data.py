import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split

import pytorch_lightning as pl
from PIL import Image

class LeftRightDaVinciDataSet(Dataset):

    def __init__(self,
                 root_dir: str,
                 image_set: str = 'train',
                 frames_per_sample: int = 5,
                 frames_to_drop: int = 2,
                 resize: float = 1,   #leaving this in case we want to add the ability to resize
                 img_transform = None,
                 target_transform = None
                 ):
        self.root_dir = root_dir
        self.image_set = image_set

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

        left_images = []
        right_images = []
        for frame in frames:
            left_img_path = os.path.join(self.root_dir, '{:s}'.format(self.image_set), 'image_0', '{:s}'.format(frame))
            right_img_path = os.path.join(self.root_dir, '{:s}'.format(self.image_set), 'image_1', '{:s}'.format(frame))

            left_image = Image.open(left_img_path)
            right_image = Image.open(right_img_path)

            l_image = self.img_transform(left_image)
            r_image = self.img_transform(right_image)

            left_images.append(l_image)
            left_images.append(r_image)

        # channels are the b/w input frames
        left_image_tensor = torch.stack(left_images)
        left_image_tensor = torch.squeeze(left_image_tensor, 1)
        right_image_tensor = torch.stack(right_images)
        right_image_tensor = torch.squeeze(right_image_tensor, 1)

        # target is only the last frame
        target_path = os.path.join(self.root_dir, '{:s}'.format(self.image_set), 'disparity', '{:s}'.format(frames[-1]))
        target = Image.open(target_path)
        target = self.target_transform(target)

        return left_image_tensor, right_image_tensor, target

ds = LeftRightDaVinciDataSet('/Users/annikabrundyn/Developer/da_vinci_depth/daVinci_data')
dm = DaVinciDataModule('/Users/annikabrundyn/Developer/da_vinci_depth/daVinci_data')