'''
@Author: Yuan Wang
@Contact: wangyuan2020@ia.ac.cn
@File: My_args.py
@Time: 2021/12/02 10:55 AM
'''


import argparse
import torch

parser = argparse.ArgumentParser(description='3D Face Landmark Detection')
# base args
parser.add_argument('--exp_name', type=str, default='Face alignment with PAConv', metavar='N', help='Name of the experiment')
parser.add_argument('--model', type=str, default='PAConv', metavar='N', choices=['EdgeConv', 'PAConv'], help='Model to use')
parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
parser.add_argument('--dataset', type=str, default='BU-3DFE', metavar='N', choices=['BU-3DFE','FRGC','FaceScape'])


# train args
parser.add_argument('--eval', type=bool, default=False, help='evaluate the model')
parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size', help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of episode to train')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')


# optimizor args
parser.add_argument('--loss', type=str, default='adaptive_wing', metavar='N', choices=['mse', 'adaptive_wing'], help='loss function to use')
parser.add_argument('--use_sgd', type=bool, default=False, help='Use SGD')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--scheduler', type=str, default='step', metavar='N', choices=['cos', 'step'], help='Scheduler to use, [cos, step]')
parser.add_argument('--weight_decay', type=float, default=0, metavar='WD', help='the weight decay of optimizor')


# data process args
parser.add_argument('--max_threshold', default=10, type=float, help='the maximum threshold of error_rate')
parser.add_argument('--sample_way', type=str, default='FPS', metavar='sw', choices=['FPS', 'Random', 'CAGQ', 'Geometric'])
parser.add_argument('--need_resample', type=bool, default=False, help='resample data with different number of points')
parser.add_argument('--seed', type=int, default=10, metavar='S', help='random seed (default: 1)')
parser.add_argument('--regression_point_num', type=int, default=10, metavar='RPN', help='the number of points in landmark regression (default: 10)')
parser.add_argument('--dataset_seed', type=int, default=1, metavar='S', help='train/test dataset random seed (default: 1)')
parser.add_argument('--num_points', type=int, default=2048, help='num of points to use')
parser.add_argument('--sigma', type=float, default=10, metavar='Sig', help='Gaussian Variance of heatmap')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=30, metavar='N', help='Num of nearest neighbors to use in PAConv')


# PAConv args
parser.add_argument('--calc_scores', type=str, default='softmax', metavar='cs', help='The way to calculate score')
parser.add_argument('--hidden', type=list, default=[[16], [16], [16], [16]], help='the hidden layers of ScoreNet')
parser.add_argument('--num_matrices', type=list, default=[8, 8, 8, 8], help='the number of weight banks')

