import pytest
import pytorch_lightning as pl

from models.right_view.dm_right_model import DepthMapRightModel
from data.right_data import RightDaVinciDataModule
from models.callbacks.right_dm_callback import RightDepthMapCallback


# TODO: dont know how to make this cleaner
DEPTH_MAP_CHECKPOINT_PATH = "/Users/annikabrundyn/Developer/da_vinci/grid_artifacts/final_dm_artifacts/grid_artifacts/m10-2drop0-color-exp0/version_0/checkpoints/epoch=155.ckpt"


@pytest.mark.parametrize("frames_per_sample,frames_to_drop,dm_model_checkpoint",
                         [(2, 0, DEPTH_MAP_CHECKPOINT_PATH)])
def test_right_model(seed_everything, data_dir, frames_per_sample, frames_to_drop, dm_model_checkpoint):
    dm = RightDaVinciDataModule(data_dir,
                                frames_per_sample=frames_per_sample,
                                frames_to_drop=frames_to_drop,
                                extra_info=True,
                                is_color_input=True,
                                is_color_output=True,
                                batch_size=2,
                                num_workers=0)
    dm.setup()

    # model
    model = DepthMapRightModel(dm_model_checkpoint, frames_per_sample, frames_to_drop)

    # train
    trainer = pl.Trainer(fast_dev_run=True, callbacks=[RightDepthMapCallback()])
    trainer.fit(model, dm)
