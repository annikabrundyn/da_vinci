import pytest
import pytorch_lightning as pl

from models.right_view.direct_right_model import DirectRightModel
from data.right_data import RightDaVinciDataModule
from deprecated.callbacks.right_callback import RightCallback


@pytest.mark.parametrize("frames_per_sample,frames_to_drop,is_color_input,is_color_output",
                         [(1, 0, False, False), (1, 0, True, False), (1, 0, True, True), (3, 1, True, True)])
def test_right_model(seed_everything, data_dir, frames_per_sample, frames_to_drop, is_color_input, is_color_output):
    dm = RightDaVinciDataModule(data_dir,
                                frames_per_sample=frames_per_sample,
                                frames_to_drop=frames_to_drop,
                                is_color_input=is_color_input,
                                is_color_output=is_color_output,
                                extra_info=True,
                                batch_size=2,
                                num_workers=0)
    dm.setup()

    # model
    model = DirectRightModel(frames_per_sample, frames_to_drop, is_color_input, is_color_output)

    # train
    trainer = pl.Trainer(fast_dev_run=True, callbacks=[RightCallback()])
    trainer.fit(model, dm)
