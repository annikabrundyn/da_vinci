from data.data import DaVinciDataSet, DaVinciDataModule


class DepthDaVinciDataSet(DaVinciDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_dir = 'disparity'


class DepthDaVinciDataModule(DaVinciDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = DepthDaVinciDataSet
