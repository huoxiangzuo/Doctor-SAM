# Doctor-SAM （ICIC 2024 Oral）
![](https://img.shields.io/github/license/huoxiangzuo/Doctor-SAM)  
This repo. is the official implementation of '**Dr-SAM: U-shape Structure Segment Anything Model for Generalizable Medical Image Segmentation**'.   
Authors: Xiangzuo Huo, Shengwei Tian, Bingming Zhou, Long Yu, Aolun Li.  

## Overview
<!-- <img width="1395" alt="drsam" src="https://github.com/huoxiangzuo/Doctor-SAM/assets/57312968/2524a914-f4c5-46c6-bf56-85b0d6ec8d1e"> -->
<img width="640" alt="drsam" src="https://github.com/huoxiangzuo/Doctor-SAM/assets/57312968/2524a914-f4c5-46c6-bf56-85b0d6ec8d1e">

## DrSAM Segmentation Qualitative Results
<!-- <img src="https://github.com/huoxiangzuo/Doctor-SAM/assets/57312968/bd81ce6d-a1df-4ab0-975f-71d604c16895" width="75%"> -->
<img width="404" alt="result" src="https://github.com/huoxiangzuo/Doctor-SAM/assets/57312968/bd81ce6d-a1df-4ab0-975f-71d604c16895">

## Run
0. Requirements:
* python3
* pytorch <= 2.0.0

1. Train:
* Run `python train_single.py`

2. Evaluate:
* Modify `parser.add_argument('--eval', default=True)` in `train_single.py`
* Run `python train_single.py`

3. Visualize:
* Modify `parser.add_argument('--visualize', default=True)` in `train_single.py`
* Run `python train_single.py`

## <a name="Models"></a>Model Checkpoints

Click the links below to download the checkpoint for the corresponding model type.

- `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

Init checkpoint can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/pretrained_checkpoint)

### Expected checkpoint

```
pretrained_checkpoint
|____sam_vit_b_maskdecoder.pth
|____sam_vit_b_01ec64.pth
|____sam_vit_l_maskdecoder.pth
|____sam_vit_l_0b3195.pth
|____sam_vit_h_maskdecoder.pth
|____sam_vit_h_4b8939.pth

```

## Reference
Some of the codes in this repo are borrowed from:  
* [SAM](https://github.com/facebookresearch/segment-anything)  
* [SAM-HQ](https://github.com/SysCV/sam-hq)
  
Thank them for their awesome work!

## Citation

If you find our paper/code is helpful, please consider citing:

```
@inproceedings{huo2024dr,
  title={Dr-SAM: U-Shape Structure Segment Anything Model for Generalizable Medical Image Segmentation},
  author={Huo, Xiangzuo and Tian, Shengwei and Zhou, Bingming and Yu, Long and Li, Aolun},
  booktitle={International Conference on Intelligent Computing},
  pages={197--207},
  year={2024},
  organization={Springer}
}
```
