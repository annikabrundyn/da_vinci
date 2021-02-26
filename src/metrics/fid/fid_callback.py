''' fid code and inception model from https://github.com/mseitzer/pytorch-fid '''

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from scipy import linalg
import pickle
import torch
import numpy as np
from tqdm import tqdm
import os
from hydra.utils import instantiate
from tqdm import tqdm

from metrics.fid.fid_components import load_patched_inception_v3, calc_fid


class FIDCallback(pl.callbacks.base.Callback):
    '''
    db_stats - pickle file with inception stats on real data
    n_samples - number of samples for FID
    '''

    def __init__(self, db_stats, dm, noise_dim=3,
                 data_transform=None, n_samples=5000, batch_size=16):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.inception = load_patched_inception_v3()
        self.noise_dim = noise_dim

        self.val_dl = dm.val_dataloader()

        if not os.path.isfile(db_stats):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("Ground Truth inception stats pickle not found.")
            print(f"Creating using device {device}")
            self.inception = self.inception.to(device)
            features = []

            batches = 0
            for i, (real_im, _) in enumerate(tqdm(self.val_dl, desc="Getting features for real data")):
                if batches > 5:
                    break

                # check whether this is first or last frame
                real_im = real_im[0]
                real_im = real_im.to(device)

                feat = self.inception(real_im)[0].view(real_im.shape[0], -1)  # compute features
                features.append(feat.to('cpu'))

                batches += 1
            features = torch.cat(features, 0).numpy()
            self.inception = self.inception.to(torch.device('cpu'))
            self.real_mean = np.mean(features, 0)
            self.real_cov = np.cov(features, rowvar=False)

            with open(db_stats, 'wb') as handle:
                pickle.dump({'mean': self.real_mean, 'cov': self.real_cov},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Load inception statistics computed on real data
        with open(db_stats, 'rb') as f:
            embeds = pickle.load(f)
            self.real_mean = embeds['mean']
            self.real_cov = embeds['cov']

    def to(self, device):
        self.inception = self.inception.to(device)

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.eval()

        with torch.no_grad():
            self.to(pl_module.device)
            features = []

            batches = 5
            for i, (img, target) in enumerate(tqdm(self.val_dl, desc="Getting features for generated images.")):
                if batches > 5:
                    break

                pred = pl_module(img)

                feat = self.inception(pred)[0].view(pred.shape[0], -1)  # compute features
                features.append(feat.to('cpu'))

                batches += 1

            features = torch.cat(features, 0)[:self.n_samples].numpy()

            sample_mean = np.mean(features, 0)
            sample_cov = np.cov(features, rowvar=False)

            fid = calc_fid(sample_mean, sample_cov, self.real_mean, self.real_cov)
            print(f"FID: {fid}\n")

            # log FID
            # pl_module.log("val_fid", fid)
            # self.to(torch.device('cpu'))

        self.last_global_step = trainer.global_step