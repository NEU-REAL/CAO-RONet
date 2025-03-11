<h1 align="center"> CAO-RONet: A Robust 4D Radar Odometry with Exploring More Information from Low-Quality Points </h1>

<div align="center">

**Zhiheng Li**, Yubo Cui, Ningyuan Huang, Chenglin Pang, Zheng Fang*

[![arXiv](https://img.shields.io/badge/arXiv-2503.01438-b31b1b.svg)](https://arxiv.org/abs/2503.01438)

</div>

## Absract

Recently, 4D millimetre-wave radar exhibits more stable perception ability than LiDAR and camera under adverse conditions (e.g. rain and fog). However, low-quality radar points hinder its application, especially the odometry task that requires a dense and accurate matching. To fully explore the potential of 4D radar, we introduce a learning-based odometry framework, enabling robust ego-motion estimation from finite and uncertain geometry information. First, for sparse radar points, we propose a local completion to supplement missing structures and provide denser guideline for aligning two frames. Then, a context-aware association with a hierarchical structure flexibly matches points of different scales aided by feature similarity, and improves local matching consistency through correlation balancing. Finally, we present a window-based optimizer that uses historical priors to establish a coupling state estimation and correct errors of inter-frame matching. The superiority of our algorithm is confirmed on View-of-Delft dataset, achieving around a 50% performance improvement over previous approaches and delivering accuracy on par with LiDAR odometry.

## Overview

<p align="center">
    <img src="images/method.png" width="100%">
</p>

The overview of our proposed CAO-RONet. At first, the two frames of radar features derived from backbone are fed into LCM to densify sparse points. Then, CAM implements feature-assisted registration to associate point pairs in different scales, followed by correlation balancing to suppress outliers. Finally, COM with sequential state modeling applies historical prior from clip window to constraint the current ego-motion prediction and smooth trajectory.

## Quickstart

### 1. Environment Setup
Our code is implemented on Python 3.8 with Pytorch 2.1.0 and CUDA 11.8. To reproduce and use our environment, you can use the following command:

a. Clone the repository to local
```
git clone https://github.com/NEU-REAL/CAO-RONet.git
cd CAO-RONet
```               
b. Set up a new environment with Anaconda
```
conda create -n ronet python=3.8
conda activate ronet
```                       
c. Install common dependices and pytorch
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


## Acknowledgement

This repo is based on [CMFlow](https://github.com/Toytiny/CMFlow), we are very grateful for their excellent work.                     
We are also very grateful for the support and assistance of Shouyi Lu, the author of [4DRO-Net](https://ieeexplore.ieee.org/document/10237296/).

## Citation

If you find our repository useful, please consider citing us as
```
@article{li2025cao,
  title={CAO-RONet: A Robust 4D Radar Odometry with Exploring More Information from Low-Quality Points},
  author={Li, Zhiheng and Cui, Yubo and Huang, Ningyuan and Pang, Chenglin and Fang, Zheng},
  journal={arXiv preprint arXiv:2503.01438},
  year={2025}
}
```
