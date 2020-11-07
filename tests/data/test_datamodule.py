import os
from src.data.data import DaVinciDataModule

DATA_DIR = "//daVinci_data"


def test_val_test_splits(request):

    dm1 = DaVinciDataModule(data_dir=DATA_DIR, frames_per_sample=1, frames_to_drop=0, batch_size=1, extra_info=True)
    dm1.setup()
    dm2 = DaVinciDataModule(data_dir=DATA_DIR, frames_per_sample=5, frames_to_drop=3, batch_size=1, extra_info=True)
    dm2.setup()

    # test deterministic val split
    assert dm1.val_sets == dm2.val_sets

    # test deterministic test split
    assert dm1.test_sets == dm2.test_sets

    # check pred images set
    assert dm1.vis_img_list[0][1][0] == dm2.vis_img_list[0][1][0]


def test_right_view_dm(request):

    dm3 = DaVinciDataModule(data_dir=DATA_DIR, frames_per_sample=2, frames_to_drop=1, batch_size=1, include_right_view=True, extra_info=False)
    dm3.setup()
    img3, target3 = next(iter(dm3.train_dataloader()))

    assert img3.shape == (1, 2, 192, 384)
    assert target3.shape == (1, 1, 178, 306)



