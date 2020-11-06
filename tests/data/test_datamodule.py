import os
from PIL import Image
from torchvision import transforms

from src.data.data import DaVinciDataModule

DATA_DIR = "/Users/annikabrundyn/Developer/da_vinci_depth/daVinci_data"

def test_val_test_splits(request):

    dm1 = DaVinciDataModule(data_dir=DATA_DIR, frames_per_sample=1, frames_to_drop=0, batch_size=1,
                            extra_info=True)
    dm1.setup()

    dm2 = DaVinciDataModule(data_dir=DATA_DIR, frames_per_sample=5, frames_to_drop=3, batch_size=1,
                            extra_info=True)
    dm2.setup()

    dm3 = DaVinciDataModule(data_dir=DATA_DIR, frames_per_sample=2, frames_to_drop=1, batch_size=1,
                            include_right_view=True, extra_info=True)
    dm3.setup()


