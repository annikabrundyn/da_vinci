import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
from inception import InceptionV3
import pickle
from scipy import linalg
from pytorch_lightning.utilities import rank_zero_only



def load_patched_inception_v3():
    inception_feat = InceptionV3([3], normalize_input=False).eval()
    return inception_feat


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    ''' https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py '''
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid





def calc_fid_pytorch(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    ''' https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py '''
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)
    cov_sqrt = torch.from_numpy(cov_sqrt)

    # check if contains complex numbers - dont know how to do in pytorch
    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    if not torch.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = torch.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = torch.trace(sample_cov) + torch.trace(real_cov) - 2 * torch.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid




class FIDCallback(pl.callbacks.base.Callback):
    '''
    db_stats - pickle file with inception stats on real data
    n_samples - number of samples for FID
    '''

    def __init__(self, db_stats, full_val_dataset, batch_size=32, n_samples=100):
        self.batch_size = batch_size
        self.inception = load_patched_inception_v3()
        self.n_samples = n_samples
        self.full_val_dataset = full_val_dataset

        # If we haven't already calculated real data stats - calculate
        if not os.path.isfile(db_stats):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("Ground Truth inception stats pickle not found.")
            print(f"Creating using device {device}")
            self.inception = self.inception.to(device)
            features = []

            # val dataset
            real_dataloader = DataLoader(self.full_val_dataset, batch_size=self.batch_size)

            total_batches = len(real_dataloader)
            for i, (real_im, _) in enumerate(tqdm(real_dataloader, desc="Getting features for real data")):
                real_im = real_im.to(device)
                if real_im.shape[1] == 1:
                    # convert greyscale to RGB
                    real_im = torch.cat(3 * [real_im], dim=1)
                feat = self.inception(real_im)[0].view(real_im.shape[0], -1)  # compute features
                features.append(feat.to('cpu'))
            features = torch.cat(features, 0).numpy()
            self.inception = self.inception.to(torch.device('cpu'))
            self.real_mean = np.mean(features, 0)
            self.real_cov = np.cov(features, rowvar=False)

            # save stats as pickle file
            with open(db_stats, 'wb') as handle:
                pickle.dump({'mean': self.real_mean, 'cov': self.real_cov},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Load real data stats - ie real_mean, real_cov
        with open(db_stats, 'rb') as f:
            embeds = pickle.load(f)
            #self.real_mean = embeds['mean']
            #self.real_cov = embeds['cov']
            self.real_mean = torch.from_numpy(embeds['mean'])
            self.real_cov = torch.from_numpy(embeds['cov'])

    def to(self, device):
        self.inception = self.inception.to(device)
        self.z_samples = [z.to(device) for z in self.z_samples]

    @rank_zero_only
    def on_validation_start(self, trainer, pl_module):
        '''
        Initialize random noise and inception module
        I keep the model and the noise on CPU when it's not needed to preserve memory; could also be initialized on pl_module.device
        '''
        noise_dim = 3 #not sure what this is
        self.z_samples = [torch.randn(self.batch_size, noise_dim) for i in
                          range(0, self.n_samples, self.batch_size)]
        print('\nFID initialized')

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        # calculate fid for validation samples
        pl_module.eval()

        with torch.no_grad():
            self.to(pl_module.device)
            features = []

            total_batches = len(self.z_samples)
            for i, z in enumerate(tqdm(self.z_samples, desc="Getting features for fake images.")):
                inputs = z
                fake = pl_module.decoder(z)
                #TODO: remove hard coding for mnist
                fake = fake.reshape(fake.shape[0], 1, 28, 28)
                if fake.shape[1] == 1:
                    # convert greyscale to RGB
                    fake = torch.cat(3 * [fake], dim=1)
                feat = self.inception(fake)[0].view(fake.shape[0], -1)  # compute features
                features.append(feat.to('cpu'))

            features = torch.cat(features, 0)[:self.n_samples].numpy()

            sample_mean = np.mean(features, 0)
            sample_cov = np.cov(features, rowvar=False)

            sample_mean = torch.from_numpy(sample_mean)
            sample_cov = torch.from_numpy(sample_cov)


            fid = calc_fid_pytorch(sample_mean, sample_cov, self.real_mean, self.real_cov)

            print(f"FID: {fid}\n")


