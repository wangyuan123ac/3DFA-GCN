"""
@Author: Yuan Wang
@Contact: wangyuan2020@ia.ac.cn
@File: augmentations.py
@Time: 2021/12/02 10:03 AM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from My_args import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''data normalization'''
def normalize_data(batch_data):
    # batch_data : batch_size * num_points * num_dims
    B, N, C = batch_data.shape
    centroid = torch.mean(batch_data, axis=1)
    batch_data = batch_data - centroid.unsqueeze(1).repeat(1, N, 1)
    m = torch.max(torch.sqrt(torch.sum(batch_data ** 2, axis=2)),axis=1)[0]
    batch_data = batch_data / m.view(-1, 1, 1)
    return batch_data


'''data Scale and Translate'''
class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().to(device)) + torch.from_numpy(xyz2).float().to(device)
        return pc


