
import torch
import torch.nn as nn
import torch.nn.functional as F
from My_args import *
from PAConv.util.PAConv_util import knn, get_graph_feature, get_scorenet_input, feat_trans_dgcnn, ScoreNet, Attention_Layer
from PAConv.cuda_lib.functional import assign_score_withk as assemble_dgcnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
PAConv model from 
Xu M, Ding R, Zhao H, et al. PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds. CVPR2021
The following module is based on https://github.com/CVMI-Lab/PAConv
'''
class PAConv(nn.Module):
    def __init__(self, args, landmark_num):
        super(PAConv, self).__init__()
        # baseline args:
        self.args = args
        # PAConv args:
        self.k = args.k
        self.landmark_num = landmark_num
        self.calc_scores = args.calc_scores
        self.hidden = args.hidden

        self.m2, self.m3, self.m4, self.m5 = args.num_matrices
        self.scorenet2 = ScoreNet(10, self.m2, hidden_unit=self.hidden[0])
        self.scorenet3 = ScoreNet(10, self.m3, hidden_unit=self.hidden[1])
        self.scorenet4 = ScoreNet(10, self.m4, hidden_unit=self.hidden[2])
        self.scorenet5 = ScoreNet(10, self.m5, hidden_unit=self.hidden[3])

        i2 = 64       # channel dim of input_2nd
        o2 = i3 = 64  # channel dim of output_2st and input_3rd
        o3 = i4 = 64  # channel dim of output_3rd and input_4th
        o4 = i5 = 64  # channel dim of output_4th and input_5th
        o5 = 64       # channel dim of output_5th

        tensor2 = nn.init.kaiming_normal_(torch.empty(self.m2, i2 * 2, o2), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i2 * 2, self.m2 * o2)
        tensor3 = nn.init.kaiming_normal_(torch.empty(self.m3, i3 * 2, o3), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i3 * 2, self.m3 * o3)
        tensor4 = nn.init.kaiming_normal_(torch.empty(self.m4, i4 * 2, o4), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i4 * 2, self.m4 * o4)
        tensor5 = nn.init.kaiming_normal_(torch.empty(self.m5, i5 * 2, o5), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i4 * 2, self.m5 * o5)

        self.matrice2 = nn.Parameter(tensor2, requires_grad=True)
        self.matrice3 = nn.Parameter(tensor3, requires_grad=True)
        self.matrice4 = nn.Parameter(tensor4, requires_grad=True)
        self.matrice5 = nn.Parameter(tensor5, requires_grad=True)

        self.bn2 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn3 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn4 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn5 = nn.BatchNorm1d(64, momentum=0.1)

        self.bnt = nn.BatchNorm1d(1024, momentum=0.1)
        self.bnc = nn.BatchNorm1d(64, momentum=0.1)

        self.bn6 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn7 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn8 = nn.BatchNorm1d(128, momentum=0.1)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),     # 6 18
                                   nn.BatchNorm2d(64, momentum=0.1))

        self.convt = nn.Sequential(nn.Conv1d(64*5, 1024, kernel_size=1, bias=False),
                                   self.bnt)

        self.conv6 = nn.Sequential(nn.Conv1d(1024+64*5, 256, kernel_size=1, bias=False),
                                   self.bn6)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn7)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv8 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn8)
        self.conv9 = nn.Conv1d(128, landmark_num, kernel_size=1, bias=True)

        
    def forward(self, x):
        B, C, N = x.size()
        idx, _ = knn(x, k=self.k)
        xyz = get_scorenet_input(x, k=self.k, idx=idx)  # ScoreNet input
        # use MLP at the 1st layer, same with DGCNN
        x = get_graph_feature(x, k=self.k, idx=idx)
        x = x.permute(0, 3, 1, 2)  # b,2cin,n,k
        x = F.relu(self.conv1(x))
        x1 = x.max(dim=-1, keepdim=False)[0]
        # replace the last 4 DGCNN-EdgeConv with PAConv:
        """CUDA implementation of PAConv: (presented in the supplementary material of the paper)"""
        """feature transformation:"""
        x2, center2 = feat_trans_dgcnn(point_input=x1, kernel=self.matrice2, m=self.m2)
        score2 = self.scorenet2(xyz, calc_scores=self.calc_scores, bias=0)
        """assemble with scores:"""
        x = assemble_dgcnn(score=score2, point_input=x2, center_input=center2, knn_idx=idx, aggregate='sum')
        x2 = F.relu(self.bn2(x))

        x3, center3 = feat_trans_dgcnn(point_input=x2, kernel=self.matrice3, m=self.m3)
        score3 = self.scorenet3(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_dgcnn(score=score3, point_input=x3, center_input=center3, knn_idx=idx, aggregate='sum')
        x3 = F.relu(self.bn3(x))

        x4, center4 = feat_trans_dgcnn(point_input=x3, kernel=self.matrice4, m=self.m4)
        score4 = self.scorenet4(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_dgcnn(score=score4, point_input=x4, center_input=center4, knn_idx=idx, aggregate='sum')
        x4 = F.relu(self.bn4(x))

        x5, center5 = feat_trans_dgcnn(point_input=x4, kernel=self.matrice5, m=self.m5)
        score5 = self.scorenet5(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_dgcnn(score=score5, point_input=x5, center_input=center5, knn_idx=idx, aggregate='sum')
        x5 = F.relu(self.bn5(x))

        xx = torch.cat((x1, x2, x3, x4, x5), dim=1)

        xc = F.relu(self.convt(xx))
        xc = F.adaptive_max_pool1d(xc, 1).view(B, -1)
        cls = xc.view(B, 1024, 1).repeat(1, 1, N)
        x = torch.cat((xx, cls), dim=1)
        x = F.relu(self.conv6(x))
        x = self.dp1(x)
        x = F.relu(self.conv7(x))
        x = self.dp2(x)
        x = F.relu(self.conv8(x))
        """ Output the heatmap regression result: """
        x = self.conv9(x) 
        return x


