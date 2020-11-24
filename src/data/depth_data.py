from data.data import DaVinciDataSet, DaVinciDataModule


class DepthDaVinciDataSet(DaVinciDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_dir = 'disparity'


class DepthDaVinciDataModule(DaVinciDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = DepthDaVinciDataSet


# dm = DepthDaVinciDataModule(data_dir = "/Users/annikabrundyn/Developer/da_vinci/daVinci_data", frames_per_sample=1, frames_to_drop=0, extra_info=True)
# dm.setup()
# i, t, e = next(iter(dm.train_dataloader()))
