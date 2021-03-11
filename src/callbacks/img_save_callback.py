import os.path

from pytorch_lightning.callbacks import Callback
from torchvision.utils import save_image


class SaveImgCallBack(Callback):

    def __init__(self, dl, epoch_logging_freq: int = 10):
        self.epoch_logging_freq = epoch_logging_freq
        self.dl = dl

    def on_validation_epoch_end(self, trainer, pl_module):

        if trainer.current_epoch % self.epoch_logging_freq == 0:
            print('save predicted val images')

            base_dir = os.path.split(trainer.checkpoint_callback.dirpath)[0]
            epoch_dir_path = os.path.join(base_dir, f"epoch_{trainer.current_epoch}")
            os.makedirs(epoch_dir_path)

            for i, (input, _) in enumerate(self.dl):
                input = input.to(pl_module.device)
                pred = pl_module(input)

                pred_path = os.path.join(epoch_dir_path, f"pred_{i}.png")
                save_image(pred, pred_path)
