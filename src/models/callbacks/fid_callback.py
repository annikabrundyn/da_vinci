import logging

from pytorch_lightning.callbacks import Callback
from metrics.fid import calculate_fid
from pytorch_fid.inception import InceptionV3


def log_fid(trainer, pl_module, dataloader, set_name, num_batches, inception_model):
    dl_iter = iter(dataloader)
    for i in range(num_batches):
        img, target, extra_info = next(dl_iter)
        pred = pl_module(img.to(pl_module.device))
        try:
            fid_val = calculate_fid(pred, target, is_color=pl_module.is_color_output, device=pl_module.device, inception_model=inception_model)
            pl_module.logger.log_metrics({f"{set_name}_fid": fid_val}, trainer.global_step)
        except ValueError:
            logging.warning('Error encountered in FID calculation!')


class FidCallback(Callback):
    def __init__(self, train_dl, valid_dl, train_epoch_freq: int = 1, valid_epoch_freq: int = 1, num_train_batches: int = 1, num_valid_batches: int = 1, dims=2048):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.train_epoch_freq = train_epoch_freq
        self.valid_epoch_freq = valid_epoch_freq
        self.num_train_batches = num_train_batches
        self.num_valid_batches = num_valid_batches

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception_model = InceptionV3([block_idx])

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if trainer.current_epoch % self.train_epoch_freq == 0:
            log_fid(trainer, pl_module, self.train_dl, "train", self.num_train_batches, self.inception_model)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.valid_epoch_freq == 0:
            log_fid(trainer, pl_module, self.valid_dl, "valid", self.num_valid_batches, self.inception_model)

    def on_fit_end(self, trainer, pl_module):
        log_fid(trainer, pl_module, self.valid_dl, "final_valid", len(self.valid_dl), self.inception_model)
