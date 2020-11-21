from pytorch_lightning.callbacks import Callback

class SavePredImgCallback(Callback):
    def __init__(self, dl, epoch_logging_freq: int = 50):
        # save every 50 epochs
        self.epoch_logging_freq = epoch_logging_freq
        self.dl = dl

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.epoch_logging_freq == 0:
            batch_idx = 0
            for img, target, extra in self.dl:
                img, target = img.to(pl_module.device), target.to(pl_module.device)
                folder_name = extra['image_set'][0]
                frame_nums = extra['frame_nums'][0]

                pred = pl_module(img)

                pl_module._matplotlib_imshow_input_imgs(img.squeeze(0), folder_name, frame_nums, save_fig=True, title=f"input_{batch_idx}", trainer=trainer)
                pl_module._matplotlib_imshow_dm(target.squeeze(0), title=f"target_{batch_idx}", save_fig=True, trainer=trainer)
                pl_module._matplotlib_imshow_dm(pred.squeeze(0), title=f"prediction_{batch_idx}", save_fig=True, trainer=trainer)

                batch_idx += 1