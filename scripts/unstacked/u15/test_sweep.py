import torch
import fire
import itertools
import numpy
from models.right_view.unstacked_unet2d import main as unstacked_main


def retreve_config(sweep_step):
    grid = {
        "lr": [0.0001],
        "loss": ["l1"],
    }

    grid_setups = list(
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    )
    step_grid = grid_setups[sweep_step - 1]  # slurm var will start from 1

    config = [
        "--lr", str(step_grid["lr"]),
        "--loss", str(step_grid["loss"]),
        "--extra_skip", "True",
        "--gpus", "1",
        "--precision", "16",
        "--batch_size", "64",
        "--default_root_dir", "/scratch/js11133/da_vinci/exps/unstacked/u15",
        "--data_dir", "/scratch/js11133/da_vinci/raw_data"
    ]

    return config


def main(sweep_step):
    config = retreve_config(sweep_step)
    print(config)
    unstacked_main(config)


if __name__ == "__main__":
    fire.Fire(main)
