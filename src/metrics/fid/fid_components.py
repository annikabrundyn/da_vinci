from scipy import linalg
from pytorch_fid.inception import InceptionV3  # https://github.com/mseitzer/pytorch-fid
import pickle
import torch
import numpy as np
from tqdm import tqdm


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
