import os
import torch
from tqdm import tqdm
from argparse import ArgumentParser

import pytorch_lightning as pl
from torchvision.utils import save_image

from models.right_view.unstacked_unet2d import UnstackedModel
from models.right_view.stacked_unet2d import StackedModel
from data.multiframe_data import UnstackedDaVinciDataModule, StackedDaVinciDataModule

from pytorch_lightning.metrics.functional import ssim, psnr
import lpips
import DISTS_pytorch


if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = UnstackedModel.add_model_specific_args(parser)
    parser.add_argument("--ckpt", required=True, type=str, help="path to model checkpoint")
    parser.add_argument("--max_frame_exp", type=int, default=10)
    parser.add_argument("--stacked", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.stacked:
        m = StackedModel
        d = StackedDaVinciDataModule
    else:
        m = UnstackedModel
        d = UnstackedDaVinciDataModule

    # model
    model = m.load_from_checkpoint(checkpoint_path=args.ckpt)
    model.to(device)
    model.eval()
    model.freeze()

    # data
    dm = d(
        args.data_dir,
        frames_per_sample=model.hparams.num_frames,
        frames_to_drop=0,
        extra_info=False,
        batch_size=args.batch_size,
        num_workers=model.hparams.num_workers,
        videos_drop_k=(args.max_frame_exp - model.hparams.num_frames),
    )
    dm.setup()
    print("dm setup")

    if args.test:
        dl = dm.test_dataloader()
    else:
        dl = dm.val_dataloader()

    print("hi \n")
    LPIPS_ALEX = lpips.LPIPS(net='alex', eval_mode=True).to(device)
    #LPIPS_VGG = lpips.LPIPS(net='vgg', eval_mode=True).to(device)
    DISTS = DISTS_pytorch.DISTS().to(device)

    lpips_alex_sum = 0
    #lpips_vgg_sum = 0
    dists_sum = 0

    ssim_avg_sum = 0
    psnr_avg_sum = 0


    for batch_idx, batch in enumerate(tqdm(dl)):

        img, target = batch
        img = img.to(device)
        target = target.to(device)
        pred = model(img)

        # calculate metrics
        lpips_alex_sum += LPIPS_ALEX(pred, target).sum().item()
        print(lpips_alex_sum)
        #lpips_vgg_sum += LPIPS_VGG(pred, target).sum()
        dists_sum += DISTS(pred, target).sum().item()

        ssim_avg_sum += ssim(pred, target)
        psnr_avg_sum += psnr(pred, target)

    # average
    final_lpips_alex = lpips_alex_sum / len(dl.dataset)
    #final_lpips_vgg = lpips_vgg_sum / len(dl.dataset)
    final_dists = dists_sum / len(dl.dataset)

    final_ssim = ssim_avg_sum / len(dl)
    final_psnr = psnr_avg_sum / len(dl)

    print("---RESULTS---")
    print("LPIPS (alex): ", final_lpips_alex)
    #print("LPIPS (vgg): ", final_lpips_vgg)
    print("DISTS: ", final_dists)
    print("SSIM: ", final_ssim)
    print("PSNR: ", final_psnr)

    print("done")

