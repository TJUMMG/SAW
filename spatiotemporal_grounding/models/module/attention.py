import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


class GlobalTextPresentation(nn.Module):
    def __init__(self, text_dim):
        super(GlobalTextPresentation, self).__init__()
        self.W_txt = nn.Linear(text_dim, text_dim)

    def forward(self, fea_text, mask=None):
        weight_text = self.W_txt(fea_text)  # B*L*C
        if mask is not None:
            weight_text = weight_text.masked_fill(mask == 0, -1e9)
        weight_text = weight_text.softmax(dim=1)
        fea_text_global = fea_text * weight_text
        fea_text_global = fea_text_global.sum(dim=1)  # B*C
        return fea_text_global


class MuTan(nn.Module):
    def __init__(self, video_fea_dim, text_fea_dim, out_fea_dim, heads=5):
        super(MuTan, self).__init__()

        self.heads = heads
        self.Wv = nn.ModuleList(
            [nn.Conv2d(video_fea_dim+8, out_fea_dim, 1, 1) for i in range(heads)])
        self.Wt = nn.ModuleList(
            [nn.Conv2d(text_fea_dim, out_fea_dim, 1, 1) for i in range(heads)])

    def forward(self, video_fea, text_fea, spatial):
        video_fea = torch.cat([video_fea, spatial], dim=1)
        fea_outs = []
        for i in range(self.heads):
            fea_v = self.Wv[i](video_fea)
            fea_v = torch.tanh(fea_v)  # B*C*H*W

            fea_t = self.Wt[i](text_fea)
            fea_t = torch.tanh(fea_t)  # B*C*1*1

            fea_out = fea_v * fea_t
            fea_outs.append(fea_out.unsqueeze(-1))
        fea_outs = torch.cat(fea_outs, dim=-1)
        fea_outs = torch.sum(fea_outs, dim=-1)
        mutan_fea = torch.tanh(fea_outs)
        mutan_fea = F.normalize(mutan_fea, dim=1)
        return mutan_fea


class RelevanceFilter(nn.Module):
    def __init__(self, text_fea_dim, video_fea_dim, attention_dim, groups=8, kernelsize=(1, 1), dilation=(1, 1), phase='3D'):
        super().__init__()
        assert phase in ['1D', '2D', '3D']
        assert text_fea_dim % groups == 0
        assert attention_dim % groups == 0
        self.phase = phase
        self.groups = groups
        self.kernel_size = kernelsize
        self.dilation = dilation
        if phase == '1D':
            assert len(kernelsize) == 1 and len(dilation) == 1
            self.Wkv = nn.Conv1d(video_fea_dim, 2*attention_dim, 1, 1)
            self.Wt = nn.Linear(text_fea_dim, attention_dim * kernelsize[0])
            self.padding = (kernelsize[0]//2)*dilation[0]
        elif phase == '2D':
            assert len(kernelsize) == 2 and len(dilation) == 2
            self.Wkv = nn.Conv2d(video_fea_dim, 2*attention_dim, 1, 1)
            self.Wt = nn.Linear(text_fea_dim, attention_dim *
                                kernelsize[0] * kernelsize[1])
            self.padding = (
                (kernelsize[0]//2)*dilation[0], (kernelsize[1]//2)*dilation[1])
        elif phase == '3D':
            assert len(kernelsize) == 3 and len(dilation) == 3
            self.Wkv = nn.Conv3d(video_fea_dim, 2*attention_dim, 1, 1)
            self.Wt = nn.Linear(text_fea_dim, attention_dim *
                                kernelsize[0] * kernelsize[1] * kernelsize[2])
            self.padding = ((kernelsize[0]//2)*dilation[0], (kernelsize[1]//2)
                            * dilation[1], (kernelsize[2]//2)*dilation[2])

    def forward(self, video_fea, text_fea, masks=None):
        b = video_fea.shape[0]

        kv = self.Wkv(video_fea)
        k, v = kv.chunk(2, dim=1)
        kernel = self.Wt(text_fea)

        if self.phase == '1D':
            kernel = repeat(kernel, 'b (g c k0) -> (b g) c k0',
                            k0=self.kernel_size[0], g=self.groups)
            k = repeat(k, 'b c l0 -> n (b c) l0', n=1)
            att = F.conv1d(k, kernel, padding=self.padding,
                           dilation=self.dilation[0], groups=b*self.groups)
            att = rearrange(att, 'n (b g c) l0 -> (n b) g c l0',
                            b=b, g=self.groups)
            v = rearrange(v, 'b (g c) l0 -> b g c l0', g=self.groups)
        elif self.phase == '2D':
            kernel = repeat(kernel, 'b (g c k0 k1) -> (b g) c k0 k1',
                            k0=self.kernel_size[0], k1=self.kernel_size[1], g=self.groups)
            k = repeat(k, 'b c l0 l1 -> n (b c) l0 l1', n=1)
            att = F.conv2d(k, kernel, padding=self.padding,
                           dilation=self.dilation, groups=b*self.groups)
            att = rearrange(
                att, 'n (b g c) l0 l1 -> (n b) g c l0 l1', b=b, g=self.groups)
            v = rearrange(v, 'b (g c) l0 l1 -> b g c l0 l1', g=self.groups)
        elif self.phase == '3D':
            kernel = repeat(kernel, 'b (g c k0 k1 k2) -> (b g) c k0 k1 k2',
                            k0=self.kernel_size[0], k1=self.kernel_size[1], k2=self.kernel_size[2], g=self.groups)
            k = repeat(k, 'b c l0 l1 l2 -> n (b c) l0 l1 l2', n=1)
            att = F.conv3d(k, kernel, padding=self.padding,
                           dilation=self.dilation, groups=b*self.groups)
            att = rearrange(
                att, 'n (b g c) l0 l1 l2 -> (n b) g c l0 l1 l2', b=b, g=self.groups)
            v = rearrange(
                v, 'b (g c) l0 l1 l2 -> b g c l0 l1 l2', g=self.groups)
        active_map = att.mean(dim=1)
        out = v * torch.sigmoid(att)
        out = torch.flatten(out, 1, 2)

        if masks is not None:
            out = out * masks
            active_map = active_map.sigmoid() * masks
        return active_map, out
