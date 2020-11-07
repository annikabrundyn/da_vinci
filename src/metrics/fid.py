import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

def get_activations(batch, model, dims=2048, device='cpu'):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- batch       :
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    pred_arr = np.empty((len(batch), dims))

    start_idx = 0

    batch = batch.to(device)

    with torch.no_grad():
        pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred = pred.squeeze(3).squeeze(2).cpu().numpy()

    pred_arr[start_idx:start_idx + pred.shape[0]] = pred

    start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_activation_statistics(batch, model, dims=2048, device='cpu'):
    """Calculation of the statistics used by the FID.
    Params:
    -- batch       :
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(batch, model, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_fid(preds, truths, is_bw=True, dims=2048, device='cpu'):
    """Calculates the FID of two paths"""
    if is_bw:
        preds = preds.repeat(1, 3, 1, 1)
        truths = truths.repeat(1, 3, 1, 1)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(preds, model,
                                         dims, device)
    m2, s2 = calculate_activation_statistics(truths, model,
                                         dims, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value