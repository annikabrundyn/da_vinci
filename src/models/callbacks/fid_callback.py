from pytorch_lightning.callbacks import Callback
from metrics.fid import calculate_fid


class FidCallback(Callback):
    def __init__(self, train_dl, valid_dl, epoch_logging_freq: int = 1):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.epoch_logging_freq = epoch_logging_freq

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        img, target, extra_info = next(iter(self.valid_dl))
        pred = pl_module(img.to(pl_module.device))
        if trainer.current_epoch % self.epoch_logging_freq == 0:
            try:
                fid_val = calculate_fid(pred, target, is_color=pl_module.is_color_output, device=pl_module.device)
            except ValueError:
                fid_val = -1
            pl_module.logger.experiment.add_scalar("train_fid", fid_val, trainer.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        img, target, extra_info = next(iter(self.valid_dl))
        pred = pl_module(img.to(pl_module.device))
        if trainer.current_epoch % self.epoch_logging_freq == 0:
            try:
                fid_val = calculate_fid(pred, target, is_color=pl_module.is_color_output, device=pl_module.device)
            except ValueError:
                fid_val = -1
            pl_module.logger.experiment.add_scalar("valid_fid", fid_val, trainer.global_step)
