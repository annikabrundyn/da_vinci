import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


class DaVinciDataSet(Dataset):
    def __init__(self,
                 data_dir: str,
                 sample_list: list,
                 frames_per_sample: int,
                 frames_to_drop: int,
                 channel_stack_frames: bool = None,
                 target_dir: str = 'image_1',
                 is_color_input: bool = True,
                 is_color_output: bool = True,
                 extra_info: bool = False,
                 ):
        self.data_dir = data_dir
        self.sample_list = sample_list
        self.frames_per_sample = frames_per_sample
        self.frames_to_drop = frames_to_drop
        self.channel_stack_frames = channel_stack_frames
        self.target_dir = target_dir

        self.is_color_input = is_color_input
        self.is_color_output = is_color_output

        self.extra_info = extra_info

        self._img_transforms()

    def _img_transforms(self):

        if self.is_color_input:
            self.img_transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.img_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

        if self.is_color_output:
            self.target_transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.target_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        image_set, frames = self.sample_list[index]

        images = []

        for frame in frames:
            img_path = os.path.join(self.data_dir, "{:s}".format(image_set), 'image_0', "{:s}".format(frame))
            image = Image.open(img_path)
            image = self.img_transform(image)
            images.append(image)

        if self.channel_stack_frames:
            image_tensor = torch.cat(images, dim=0)
        else:
            image_tensor = torch.stack(images)

        # target is only the first frame
        target_path = os.path.join(self.data_dir, "{:s}".format(image_set), self.target_dir, "{:s}".format(frames[0]))
        target = Image.open(target_path)
        target = self.target_transform(target)

        if self.extra_info:
            target_frame_name = str(frames[0])
            return image_tensor, target, target_frame_name
        else:
            return image_tensor, target