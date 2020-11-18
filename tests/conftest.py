import pytest
import pytorch_lightning as pl


def pytest_addoption(parser):
    parser.addoption(
        "--data_dir",
        required=True,
        action="store"
    )


@pytest.fixture()
def data_dir(request):
    return request.config.getoption("--data_dir")


@pytest.fixture
def seed_everything():
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(42)
