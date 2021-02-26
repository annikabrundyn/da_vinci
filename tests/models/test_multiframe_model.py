import pytest
import pytorch_lightning as pl

from models.right_view.multiframe import MultiFrameModel
from data.multiframe_data import MFDaVinciDataModule


@pytest.mark.parametrize("num_frames,combine_fn,loss,num_layers,bilinear",
                         [(3, "conv_3d", "l1", 5, True),
                          (2, "max", "perceptual", 5, False),
                          (4, "average", "l1_ssim", 6, False)])
def test_right_model(seed_everything, data_dir, num_frames, combine_fn, loss, num_layers, bilinear):

    dm = MFDaVinciDataModule(
        data_dir,
        frames_per_sample=num_frames,
        frames_to_drop=0,
        is_color_input=True,
        is_color_output=True,
        extra_info=False,
        batch_size=2,
        num_workers=0)

    dm.setup()

    # model
    model = MultiFrameModel(num_frames, combine_fn, loss, num_layers, bilinear)

    # train
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, dm)
