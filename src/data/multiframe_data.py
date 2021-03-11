from data.dataset import DaVinciDataSet
from data.datamodule import DaVinciDataModule


class StackedDaVinciDataSet(DaVinciDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channel_stack_frames = True


class StackedDaVinciDataModule(DaVinciDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = StackedDaVinciDataSet


class UnstackedDaVinciDataSet(DaVinciDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channel_stack_frames = False


class UnstackedDaVinciDataModule(DaVinciDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = UnstackedDaVinciDataSet
