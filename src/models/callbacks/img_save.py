import os.path
from pytorch_lightning.callbacks import Callback


class SaveImgCallBack(Callback):
    # for depth map models

    def __init__(self, dl, epoch_logging_freq: int = 30):
        # save predictions every 50 epochs
        self.epoch_logging_freq = epoch_logging_freq
        self.dl = dl

    def on_sanity_check_end(self, trainer, pl_module):
        '''save the input + target images only once'''
        print("do this only once")
        batch_idx = 0
        for img, target, extra in self.dl:
            folder_name = extra['image_set'][0]
            frame_nums = extra['frame_nums'][0]

            dir = trainer.checkpoint_callback.dirpath
            dir = os.path.split(dir)[0]
            inputs_dir_path = os.path.join(dir, "inputs")
            targets_dir_path = os.path.join(dir, "targets")

            pl_module._matplotlib_imshow_input_imgs(img.squeeze(0), folder_name, frame_nums, save_fig=True, title=f"input_{batch_idx}", dir_path=inputs_dir_path)
            pl_module._matplotlib_imshow_dm(target.squeeze(0), title=f"target_{batch_idx}", save_fig=True, dir_path=targets_dir_path)

            batch_idx += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        '''save only the predicted '''

        if trainer.current_epoch % self.epoch_logging_freq == 0:
            print('save predicted val images')
            batch_idx = 0
            for img, target, extra in self.dl:
                img, target = img.to(pl_module.device), target.to(pl_module.device)
                pred = pl_module(img)

                dir = os.path.split(trainer.checkpoint_callback.dirpath)[0]
                dir_path = os.path.join(dir, f"epoch_{trainer.current_epoch + 1}", "pred")

                pl_module._matplotlib_imshow_dm(pred.squeeze(0), title=f"prediction_{batch_idx}", save_fig=True, dir_path=dir_path)
                batch_idx += 1
