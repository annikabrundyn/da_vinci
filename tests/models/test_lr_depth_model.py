import pytest
import pytorch_lightning as pl
from data.depth_data import DepthDaVinciDataModule
from models.callbacks.save_pred_img_callback import SavePredImgCallback
from models.callbacks.fid_callback import FidCallback
from models.depth_map.upperbound_model import UpperBoundModel, UpperBoundImgCallback


@pytest.mark.parametrize("frames_per_sample,frames_to_drop,is_color_input",
                         [(1, 0, False), (1, 0, True)])
def test_lr_depth_model(seed_everything, data_dir, frames_per_sample, frames_to_drop, is_color_input):
    dm = DepthDaVinciDataModule(data_dir,
                                frames_per_sample=frames_per_sample,
                                frames_to_drop=frames_to_drop,
                                include_right_view=True,
                                is_color_input=is_color_input,
                                extra_info=True,
                                batch_size=2,
                                num_workers=0)
    dm.setup()

    # model
    model = UpperBoundModel(frames_per_sample=frames_per_sample, frames_to_drop=frames_to_drop, is_color_input=is_color_input)

    # train
    trainer = pl.Trainer(fast_dev_run=True, callbacks=[UpperBoundImgCallback(dm.vis_img_dataloader())])
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
