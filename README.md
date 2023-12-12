# Image Restoration - DIV2K
>*NOTE* This repo is based on this *Single Image Super Resolution Repo* [code](https://github.com/krasserm/super-resolution)

## Introduction
This repo trains models to do image restoration on DIV2K dataset. It supports all modes of distortion in DIV2K dataset including `bicubic`, `unknown`, `mild`, and `difficult`. It also restores different downscaling factors `2`, `3`, `4`, and `8`. The parameters of the training can be specified so that the model trains on a dataset, and is evaluated on a different one.


## Environment setup

create a virtual environment
```
python3 -m venv .venv
```
Activate the virtual environment
```
source .venv/bin/activate
```
Install the dependencies 
```
pip install -r requirements.txt
```

## DIV2K dataset

For training and validation on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) images, applications should use the 
provided `DIV2K` data loader. It automatically downloads DIV2K images to `.div2k` directory and converts them to a 
different format for faster loading.

A `DIV2K` [data provider](#div2k-dataset) automatically downloads [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 
training and validation images of given scale (2, 3, 4 or 8) and downgrade operator ("bicubic", "unknown", "mild" or 
"difficult"). 

### Download Training dataset

```python
from data import DIV2K

train_loader = DIV2K(scale=4,             # 2, 3, 4 or 8
                     downgrade='bicubic', # 'bicubic', 'unknown', 'mild' or 'difficult' 
                     subset='train')      # Training dataset are images 001 - 800
                     
# Create a tf.data.Dataset          
train_ds = train_loader.dataset(batch_size=16,         # batch size as described in the EDSR and WDSR papers
                                random_transform=True, # random crop, flip, rotate as described in the EDSR paper
                                repeat_count=None)     # repeat iterating over training images indefinitely
```

### Download Validation dataset

```python
from data import DIV2K

valid_loader = DIV2K(scale=4,             # 2, 3, 4 or 8
                     downgrade='bicubic', # 'bicubic', 'unknown', 'mild' or 'difficult' 
                     subset='valid')      # Validation dataset are images 801 - 900
                     
# Create a tf.data.Dataset          
valid_ds = valid_loader.dataset(batch_size=1,           # use batch size of 1 as DIV2K images have different size
                                random_transform=False, # use DIV2K images in original size 
                                repeat_count=1)         # 1 epoch
                 
```

## Train Image Restoration models

The repo supports two model architectures.

- Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)
- Wide Activation for Efficient and Accurate Image Super-Resolution (WDSR)

To run training use:

```
python train_image_resolution_model.py --scale 2
                                       --downgrade bicubic 
                                       --model edsr
                                       --batch-size 16
                                       --depth 16
                                       --scale_val 4
                                       --downgrade_val mild
```
- `model`: Model architecture, can be `edsr` or `wdsr`
- `downgrade`: Distortion type for training, can be `bicubic`, `unknown`, `mild`, or `difficult`
- `scale`: Downsampling factor for training, can be `2`, `3`, `4`, or `8`
- `depth`: Depth of the model, default `16`
- `batch-size`: Training batch size, default `16`
- `downgrade_val`: Distortion type for validation, can be `bicubic`, `unknown`, `mild`, or `difficult`
- `scale_val`: Downsampling factor for validation, can be `2`, `3`, `4`,`8`

The result model is stored under `weight/` directory.

## Baselines
| Model | Distortion | Downscaling Factor |  PSNR  | Precision |
|:-----:|:----------:|:------------------:|:------:|:---------:|
|  EDSR |   Bicubic  |          2         |        |    FP32   |
|  EDSR |   Bicubic  |          4         |  26.98 |    FP32   |
|  EDSR |   Bicubic  |          8         |        |    FP32   |
|  EDSR |    Mild    |          4         |        |    FP32   |
|  EDSR |  Difficult |          4         |        |    FP32   |
|  EDSR |   Unknown  |          2         |        |    FP32   |
|  EDSR |    All     |         All        |        |    FP32   |
