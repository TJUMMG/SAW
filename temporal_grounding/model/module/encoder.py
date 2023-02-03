import torch
from torch import nn
import numpy as np
import math, copy, time
import torch.nn.functional as F

def knn(x, y=None, k=5):
    if y is None:
        y = x
    inner = -2 * torch.matmul(y.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1)
    return idx


def get_graph_feature(x, prev_x=None, k=5, idx_knn=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx_knn is None:
        idx_knn = knn(x=x, y=prev_x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx_knn.shape[-1]
    idx_base = torch.arange(0, batch_size, device=x.device ).view(-1, 1, 1) * num_points
    idx = (idx_knn + idx_base).view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    return feature


class GCNeXtBlock(nn.Module):
    def __init__(self, channel_in, channel_out, k=3, groups=32, width_group=4):
        super(GCNeXtBlock, self).__init__()
        self.k = k
        width = width_group * groups
        self.tconvs = nn.Sequential(
            nn.Conv1d(channel_in, width, kernel_size=1), nn.ReLU(True),
            nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1), nn.ReLU(True),
            nn.Conv1d(width, channel_out, kernel_size=1),
        ) # temporal graph

        self.sconvs = nn.Sequential(
            nn.Conv2d(channel_in * 2, width, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=(1,self.k), groups=groups,  padding=(0,(self.k-1)//2)), nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        ) # semantic graph

        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x  # residual
        tout = self.tconvs(x)  # conv on temporal graph

        x_f = get_graph_feature(x, k=self.k)
        sout = self.sconvs(x_f)  # conv on semantic graph
        sout = sout.max(dim=-1, keepdim=False)[0]

        out = tout + 2 * identity + sout
        return self.relu(out)


class GCNeXtMoudle(nn.Module):
    def __init__(self, channel_in, channel_out, k_num, groups, width_group):
        super(GCNeXtMoudle, self).__init__()

        self.backbone = nn.Sequential(
            GCNeXtBlock(channel_in, channel_out, k_num, groups, width_group),
        )

    def forward(self, x):
        gcnext_feature = self.backbone(x)
        return gcnext_feature