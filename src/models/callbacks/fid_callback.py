from pytorch_lightning.callbacks import Callback
from metrics.fid import calculate_fid
from pytorch_fid.inception import InceptionV3


class FidCallback(Callback):
    def __init__(self, train_dl, valid_dl, epoch_logging_freq: int = 1):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.epoch_logging_freq = epoch_logging_freq

    def on_pretrain_routine_end(self, trainer, pl_module):
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

        self.model = InceptionV3([block_idx]).to(pl_module.device)


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch % self.epoch_logging_freq != 0:
            return

        img, target, extra_info = batch
        pred = pl_module(img)

        if batch_idx < 5:
            try:
                fid_val = calculate_fid(pred, target, self.model, is_color=pl_module.is_color_output, device=pl_module.device)
            except ValueError:
                fid_val = -1
            pl_module.log("train_fid", fid_val)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch % self.epoch_logging_freq != 0:
            return

        img, target, extra_info = batch
        pred = pl_module(img)

        # TODO: make this random
        if batch_idx < 5:
            try:
                fid_val = calculate_fid(pred, target, self.model, is_color=pl_module.is_color_output,
                                        device=pl_module.device)
            except ValueError:
                fid_val = -1
            pl_module.log("val_fid", fid_val)

