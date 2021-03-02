import pytest
import pytorch_lightning as pl
from data.depth_data import DepthDaVinciDataModule
from deprecated.callbacks import SaveImgCallBack
from deprecated.depth_map.depth_model import DepthModel


@pytest.mark.parametrize("frames_per_sample,frames_to_drop",
                         [(1, 0), (3, 1)])
def test_depth_model(seed_everything, data_dir, frames_per_sample, frames_to_drop):
    dm = DepthDaVinciDataModule(data_dir,
                                frames_per_sample=frames_per_sample,
                                frames_to_drop=frames_to_drop,
                                extra_info=True,
                                batch_size=2,
                                num_workers=0)
    dm.setup()

    # model
    model = DepthModel(frames_per_sample, frames_to_drop)

    # train
    trainer = pl.Trainer(fast_dev_run=True, callbacks=[SaveImgCallBack(dm.vis_img_dataloader())])
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
