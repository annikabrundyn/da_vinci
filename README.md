# DaVinci Depth Map + Stereo View Prediction

# Overview

### data.py

This is where the DaVinciDataModule (dm) is defined. 
* [Datamodules](https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html) are a PyTorch Lightning feature that basically allow you to specify all the data preprocessing steps up
front to ensure reproducibility in your dataset splits, transforms etc across different experiments.

**DaVinciDataModule:**

* runs setup() method to:
    * split train and test into sections of 1k consecutive frames
    * shuffle all sets and assign sets of frames to train/val/test up front
    * apply sliding window (with frame dropping) to create the actual samples within train/val/test
    * shuffle all the samples within each of the splits
    * create PyTorch Dataset from the samples to be used in the data loaders
* the train_dataloader(), val_dataloader() and test_dataloader() just put the 
datasets created in setup() into DataLoaders.
 
### model.py

This is the left view only model. To run this you have to specify:
* ```data_dir```: path to where the data is stored
* ```frames_per_sample```: number of frames to use in each sliding window
* ```frames_to_drop```: how many frames to drop within each sliding window 

Some other optional arguments:
* ```batch_size```
* ```lr```: learning rate
* ```output_img_freq```: output predicted images to tensorbaord every x batches (default 100)
* ```num_classes```: basically channels in the output (in our case 1 for grayscale depth maps)
* ```bilinear```, ```features_start```, ```num_layers``` for UNet architecture - haven't been changing these


### lr_model.py

This is the upper bound model that uses both left + right view as input
Basically exactly the same as the left only model except the input channels to the UNet is x2. 
Should really just add this as an optional parameter to the model script - will do sometime today.

### unet.py

This is the UNet architecture that is used in the two different models

# To train the left only view model
e.g. using 1 gpu

```python
python model.py --data_dir '/Users/annikabrundyn/Developer/da_vinci_depth/daVinci_data' --gpus 1 --frames_per_sample 5 --frames_to_drop 2 --batch_size 16 --lr 0.001
```

# Generating "ground truth" disparity maps

```bash
python generate_disparity.py --left_dir $LEFT_DIR --right_dir $RIGHT_DIR --output_dir $OUTPUT_DIR
```