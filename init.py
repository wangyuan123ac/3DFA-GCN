'''
@Author: Yuan Wang
@Contact: wangyuan2020@ia.ac.cn
@File: init.py
@Time: 2021/12/02 10:57 AM
'''

import os
import torch


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp My_main.py checkpoints'+'/'+args.exp_name+'/'+'My_main.py.backup')
    os.system('cp My_model.py checkpoints' + '/' + args.exp_name + '/' + 'My_model.py.backup')
    os.system('cp My_util.py checkpoints' + '/' + args.exp_name + '/' + 'My_util.py.backup')
    os.system('cp My_data.py checkpoints' + '/' + args.exp_name + '/' + 'My_data.py.backup')
    os.system('cp My_loss.py checkpoints' + '/' + args.exp_name + '/' + 'My_loss.py.backup')
    os.system('cp My_args.py checkpoints' + '/' + args.exp_name + '/' + 'My_args.py.backup')


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)