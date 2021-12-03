import torch
import numpy as np
from torch.utils.data import Dataset


def load_face_data(data):
    Heat_data_sample = np.load('./%s-npy/Heat_data_sample.npy' % data, allow_pickle=True)
    Shape_sample = np.load('./%s-npy/shape_sample.npy' % data, allow_pickle=True)
    landmark_position_select_all = np.load('./%s-npy/landmark_position_select_all.npy' % data, allow_pickle=True)
    if data == 'BU-3DFE' or data == 'FaceScape' or data == 'FRGC':
        return Shape_sample, landmark_position_select_all, Heat_data_sample


class FaceLandmarkData(Dataset):
    def __init__(self, partition='trainval', data='BU-3DFE'):
        if data == 'BU-3DFE' or data == 'FaceScape':
            self.data, self.landmark, self.seg = load_face_data(data)
        if data == 'FRGC':
            self.data, self.landmark, self.seg = load_face_data(data)
        self.partition = partition
        self.DATA = data

    def __getitem__(self, item):
        if self.DATA == 'BU-3DFE' or self.DATA == 'FaceScape':
            data_T, landmark_T, seg_T = torch.Tensor(self.data), torch.Tensor(self.landmark), torch.Tensor(self.seg)
            face = data_T[item]
        if self.DATA == 'FRGC':
            data_T, landmark_T, seg_T = torch.Tensor(self.data), torch.Tensor(self.landmark), torch.Tensor(self.seg)
            face = data_T[item]
        landmark = landmark_T[item]
        heatmap = seg_T[item]
        if self.partition == 'trainval':
            indices = list(range(face.size()[0]))
            np.random.shuffle(indices)
            face = face[indices]
            heatmap = heatmap[indices]
        return face, landmark, heatmap

    def __len__(self):
        return np.array(self.data).shape[0]




