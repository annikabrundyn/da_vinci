# DaVinci Depth Map + Stereo View Prediction

# Overview

### data.py

This is where the DaVinciDataModule (dm) is defined. 
* datamodules are a PyTorch Lightning feature that basically allow you to specify all the data preprocessing steps up
front to ensure reproducibility in your dataset splits, transforms etc across different experiments.

DaVinciDataModule:
* runs setup() method to:
    * split train and test into sections of 1k consecutive frames
    * shuffle all sets and assign sets of frames to train/val/test up front
    * apply sliding window (with frame dropping) to create the actual samples within train/val/test
    * shuffle all the samples within each of the splits
    * create PyTorch Dataset from the samples to be used in the data loaders
* the train_dataloader(), val_dataloader()
 
### model.py

This is the left view only model


### lr_model.py

This is the upper bound model that uses both left + right view as input


### unet.py

This is just the UNet architecture that is used in the two different models