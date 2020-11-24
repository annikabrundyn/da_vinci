import os.path
from pytorch_lightning.callbacks import Callback
import torchvision


class RightDepthMapCallback(Callback):
    def __init__(self, save_img_freq: int = 50):
        # save predictions every 50 epochs
        self.save_img_freq = save_img_freq

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

        for img, target, _ in trainer.datamodule.vis_img_dataloader():
            img = img[:, 0: 3,:,:]
            img = img.squeeze(0)
            target = target.squeeze(0)

            fp_input = os.path.join(inputs_dir_path, f"input_{batch_idx}.png")
            fp_target = os.path.join(targets_dir_path, f"target_{batch_idx}.png")



            torchvision.utils.save_image(img, fp=fp_input)
            torchvision.utils.save_image(target, fp=fp_target)

            batch_idx += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        '''save only the predicted '''
        if (trainer.current_epoch + 1) % self.save_img_freq == 0:
            curr_epoch_path = os.path.join(self.preds_dir_path, f"epoch_{trainer.current_epoch}")
            if not os.path.exists(curr_epoch_path):
                os.makedirs(curr_epoch_path)

            batch_idx = 0
            for img, _, _ in trainer.datamodule.vis_img_dataloader():
                img = img.to(pl_module.device)
                pred_right, pred_dm = pl_module(img)

                fp_right = os.path.join(curr_epoch_path, f"pred_right_{batch_idx}.png")
                torchvision.utils.save_image(pred_right.squeeze(0), fp=fp_right)

                fp_dm = os.path.join(curr_epoch_path, f"pred_dm_{batch_idx}.png")
                torchvision.utils.save_image(pred_dm.squeeze(0), fp=fp_dm)
                batch_idx += 1
