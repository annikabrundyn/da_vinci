import multiprocessing
import numpy as np
import pytorch_lightning as pl
import sys

from functools import partial
from tqdm import tqdm

from argparse import ArgumentParser
from data.baseline_dataset import BaselineDaVinciDataModule
from torch.utils.data import DataLoader


def mean_abs_mismatch(slice0, slice1):
    """ Mean absoute difference between images
    """
    return np.mean(np.abs(slice0 - slice1))


def x_trans_slice(img_slice, x_vox_trans):
    """ Return copy of `img_slice` translated by `x_vox_trans` voxels

    Parameters
    ----------
    img_slice : array shape (H, W, num_channels)
        2D image to transform with translation `x_vox_trans`
    x_vox_trans : int
        Number of pixels (voxels) to translate `img_slice`; can be
        positive or negative.

    Returns
    -------
    img_slice_transformed : array shape (M, N)
        2D image translated by `x_vox_trans` pixels (voxels).
    """
    # Make a 0-filled array of same shape as `img_slice`
    trans_slice = np.zeros(img_slice.shape)
    # Use slicing to select voxels out of the image and move them
    # up or down on the first (x) axis
    if x_vox_trans < 0:
        trans_slice[:, :x_vox_trans, :] = img_slice[:, -x_vox_trans:, :]
    elif x_vox_trans == 0:
        trans_slice[:, :, :] = img_slice
    else:
        trans_slice[:, x_vox_trans:, :] = img_slice[:, :-x_vox_trans, :]
    return trans_slice


def calc_mae_translation(img, target, mae_list, args):
    index = args[0]
    translation = args[1]

    img = img.numpy()
    target = target.numpy()
    img = img.transpose(1, 2, 0)
    target = target.transpose(1, 2, 0)

    # Make the translated image Y_t
    shifted = x_trans_slice(img, translation)
    # Calculate the mismatch
    mae = mean_abs_mismatch(shifted, target)
    mae_list[index] += mae


class BaselineModel:
    INPUT_IMG_WIDTH = 384

    def __init__(self):
        self.global_disparity = 0

    def fit(self, dataloader: DataLoader):
        half_width = int(self.INPUT_IMG_WIDTH / 2)
        translations = range(-half_width, half_width)  # Candidate values for t

        manager = multiprocessing.Manager()
        mae_list = manager.list([0] * len(translations))
        p = multiprocessing.Pool()

        with p:
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                imgs, targets = batch
                for img, target in zip(imgs, targets):
                    func = partial(calc_mae_translation, img, target, mae_list)
                    p.map(func, enumerate(translations))
        p.close()
        p.join()

        mae_list = np.array(mae_list)
        mae = mae_list / len(dataloader.dataset)
        min_mae_pos = mae.argmin()
        self.global_disparity = translations[min_mae_pos]
        return mae[min_mae_pos]


if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="path to davinci data")

    args = parser.parse_args()

    # data
    dm = BaselineDaVinciDataModule(args.data_dir)
    dm.setup()
    print("dm setup")

    # sanity check
    print("size of trainset:", len(dm.train_samples))
    print("size of validset:", len(dm.val_samples))
    print("size of testset:", len(dm.test_samples))

    img, target = next(iter(dm.val_dataloader()))
    print(img.shape)
    print(target.shape)

    # model
    model = BaselineModel()
    print("model instance created")

    # train
    min_mae = model.fit(dm.val_dataloader())
    global_disparity = model.global_disparity

    with open("baseline_results.txt", "w") as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(f"min mae: {min_mae}")
        print(f"best global disparity: {global_disparity}")
