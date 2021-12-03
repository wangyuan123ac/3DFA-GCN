'''
@Author: Yuan Wang
@Contact: wangyuan2020@ia.ac.cn
@File: train.py
@Time: 2021/12/02 09:59 AM
'''

import os
import math
import time
import numpy as np
from scipy import io
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from init import *
from My_args import *
from augmentations import *
from dataset import FaceLandmarkData
from loss import AdaptiveWingLoss
from util import main_sample
from PAConv_model import PAConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    writer = SummaryWriter('runs/3D_face_alignment')
    if args.need_resample:
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.dataset)
    # Dataset Random partition
    FaceLandmark = FaceLandmarkData(partition='trainval', data=args.dataset)
    train_size = int(len(FaceLandmark) * 0.7)
    test_size = len(FaceLandmark) - train_size
    torch.manual_seed(args.dataset_seed)
    # Prepare the dateset and dataloader 
    train_dataset, test_dataset = torch.utils.data.random_split(FaceLandmark, [train_size, test_size])
    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
    # data argument
    ScaleAndTranslate = PointcloudScaleAndTranslate()
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5

    # select a model to train
    model = PAConv(args, 8).to(device)   # 68 in FaceScape; 8 in BU-3DFE and FRGC
    model.apply(weight_init)
    model = nn.DataParallel(model)

    print('let us use', torch.cuda.device_count(), 'GPUs')
    if args.loss == 'adaptive_wing':
        criterion = AdaptiveWingLoss()
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0.0001)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=40, gamma=0.9)

    loss_epoch = 0.0
    for epoch in range(args.epochs):
        iters = 0
        model.train()
        for point, landmark, seg in train_loader:
            seg = torch.where(torch.isnan(seg), torch.full_like(seg, 0), seg)
            iters = iters + 1
            if args.no_cuda == False:
                point = point.to(device)                   # point: (Batch * num_point * num_dim)
                landmark = landmark.to(device)             # landmark : (Batch * landmark * num_dim)
                seg = seg.to(device)                       # seg: (Batch * point_num * landmark)
            point_normal = normalize_data(point)           # point_normal : (Batch * num_point * num_dim)
            point_normal = ScaleAndTranslate(point_normal)
            opt.zero_grad()
            point_normal = point_normal.permute(0, 2, 1)   # point : (batch * num_dim * num_point)
            pred_heatmap = model(point_normal)

            # Compute the loss fucntion 
            loss = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
            loss.backward()
            loss_epoch = loss_epoch + loss
            opt.step()
            print('Epoch: [%d / %d] Train_Iter: [%d /%d] loss: %.4f' % (epoch + 1, args.epochs, iters, len(train_loader), loss))
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), './checkpoints/%s/%s/models/model_epoch_%d.t7' % (args.exp_name, args.dataset, epoch+1))
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        writer.add_scalar('3D_Face_Alignment_loss', loss_epoch / ((epoch + 1) * len(train_loader)), epoch + 1)


if __name__ == "__main__":
    # Training settings
    args = parser.parse_args()
    _init_()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    train(args)




