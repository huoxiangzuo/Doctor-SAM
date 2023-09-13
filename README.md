# Doctor-SAM
![](https://img.shields.io/github/license/huoxiangzuo/Doctor-SAM)  
Dr. SAM. He is a fine-tuned model for medical image segmentation based on SAM.

This repo. is the official implementation of '**DrSAM: Modified Segmant Anythiny Model for Generalizable Medical Image Segmentation**'.   
Authors: Xiangzuo Huo, Shengwei Tian, Bingming Zhou, Long Yu, Aolun Li.  

## Overview
<!-- <img width="1395" alt="figure1" src="https://user-images.githubusercontent.com/57312968/191570017-34f30c13-9d8e-4776-a118-de968aebdb19.png" width="80%"> -->

## DrSAM Segmentation Quantitative Results
<!-- <img width="1424" alt="figure2s" src="https://user-images.githubusercontent.com/57312968/191570496-c62e04dc-8baf-4b01-a6ba-03c24c5a744d.png" width="70%"> -->

## DrSAM Segmentation Qualitative Results
<!-- <img src="https://user-images.githubusercontent.com/57312968/191570242-4425944d-4017-45c6-a3f7-f977376766a2.png" width="75%"> -->

## Run
0. Requirements:
* python3
* pytorch <= 2.0.0

1. Train:
* Prepare the required images and store them in categories, set up training image folders and validation image folders respectively
* Run `python train_single.py`

2. Evaluate:
* Modify `parser.add_argument('--eval', default=True)` in `train_single.py`
* Run `python train_single.py`

3. Visualize:
* Modify `parser.add_argument('--visualize', default=True)` in `train_single.py`
* Run `python train_single.py`

## <a name="Models"></a>Model Checkpoints

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)


## Reference
Some of the codes in this repo are borrowed from:  
* [SAM](https://github.com/facebookresearch/segment-anything)  
* [SAM-HQ](https://github.com/SysCV/sam-hq) 
Thank them for their awesome work!

## Citation

If you find our paper/code is helpful, please consider citing:

```bibtex

```
