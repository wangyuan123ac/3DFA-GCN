# Learning to detect 3D facial landmarks via heatmap regression with Graph Convolutional  Network

by Yuan Wang, Min Cao, Silong Peng and Zhenfeng Fan*

## Introduction

This repository is built for the official implementation of:

Learning to detect 3D facial landmarks via heatmap regression with Graph Convolutional  Network  **(AAAI2022)**

## Overview

Our 3D face alignment model is shown as follows: 

<img src="figure\3DFA-GCN.png" alt="3DFA-GCN" style="zoom: 67%;" />

We train our model on three publicly available datasets, include **BU-3DFE** (Yin et al. 2006), **FRGCv2.0** (Phillips et al. 2005) and **FaceScape** (Yang et al. 2020) to demonstrate the effectiveness of the proposed method. Our proposed method achieves **15.1%** and **10.6%** improvements in terms of the average ME on the BU-3DFE dataset and FRGCv2 dataset respectively. The ME and Std scores by the proposed method reach **1.60** and **1.18** on FaceScape dataset, respectively.

![3DFA-GCN](figure\3DFA-GCN.png)

![image-20211203204118015](C:\Users\yuawang\AppData\Roaming\Typora\typora-user-images\image-20211203204118015.png)

## Requirements:

- [x] Python 3.6.10
- [x] PyTorch 1.6.0 && torchvision 0.7.0
- [x] scikit-learn 0.23.2

## Acknowledgement

Our code base is partially borrowed from [PAConv](https://github.com/CVMI-Lab/PAConv), [DGCNN](https://github.com/WangYueFt/dgcnn) and [PointNet++](https://github.com/charlesq34/pointnet2).

