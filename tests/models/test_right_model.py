import pytest
import pytorch_lightning as pl
from data.right_data import RightDaVinciDataModule
from models.callbacks.save_pred_img_callback import SavePredImgCallback
from models.right_model import RightModel


@pytest.mark.parametrize("frames_per_sample,frames_to_drop",
                         [(1, 0)])
def test_right_model(seed_everything, data_dir, frames_per_sample, frames_to_drop):
    dm = RightDaVinciDataModule(data_dir,
                                frames_per_sample=frames_per_sample,
                                frames_to_drop=frames_to_drop,
                                extra_info=True,
                                batch_size=16,
                                num_workers=0)
    dm.setup()

    # model
    model = RightModel(frames_per_sample, frames_to_drop)

    # train
    trainer = pl.Trainer(fast_dev_run=True, callbacks=[SavePredImgCallback(dm.vis_img_dataloader())])
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
