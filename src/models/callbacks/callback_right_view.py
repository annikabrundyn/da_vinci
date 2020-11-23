import os.path
from pytorch_lightning.callbacks import Callback
import torchvision


class RightCallback(Callback):
    def __init__(self, dl, epoch_logging_freq: int = 50):
        # save predictions every 50 epochs
        self.epoch_logging_freq = epoch_logging_freq
        self.dl = dl

    def on_pretrain_routine_end(self, trainer, pl_module):
        '''save the input + target images only once'''
        print("do this only once")

        dir = os.path.split(trainer.checkpoint_callback.dirpath)[0]
        inputs_dir_path = os.path.join(dir, "inputs")
        targets_dir_path = os.path.join(dir, "targets")
        self.preds_dir_path = os.path.join(dir, "preds")

        os.makedirs(inputs_dir_path)
        os.makedirs(targets_dir_path)
        os.makedirs(self.preds_dir_path)

        batch_idx = 0

        for img, target, _ in self.dl:
            fp_input = os.path.join(inputs_dir_path, f"input_{batch_idx}.png")
            fp_target = os.path.join(targets_dir_path, f"target_{batch_idx}.png")

            torchvision.utils.save_image(img.squeeze(0), fp=fp_input)
            torchvision.utils.save_image(target.squeeze(0), fp=fp_target)

            batch_idx += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        '''save only the predicted '''
        if (trainer.current_epoch + 1) % self.epoch_logging_freq == 0:
            curr_epoch_path = os.path.join(self.preds_dir_path, f"epoch_{trainer.current_epoch}")
            if not os.path.exists(curr_epoch_path):
                os.makedirs(curr_epoch_path)

            batch_idx = 0
            for img, _, _ in self.dl:
                img = img.to(pl_module.device)
                pred = pl_module(img)
                fp = os.path.join(curr_epoch_path, f"prediction_{batch_idx}.png")
                torchvision.utils.save_image(pred.squeeze(0), fp=fp)
                batch_idx += 1
