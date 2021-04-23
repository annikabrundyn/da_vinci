from deprecated.data import DaVinciDataSet, DaVinciDataModule


class RightDaVinciDataSet(DaVinciDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_dir = 'image_1'


class RightDaVinciDataModule(DaVinciDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = RightDaVinciDataSet
