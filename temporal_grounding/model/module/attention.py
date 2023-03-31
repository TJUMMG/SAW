import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum
import warnings
warnings.filterwarnings("ignore")


class GlobalTextPresentation(nn.Module):
    def __init__(self, text_dim):
        super(GlobalTextPresentation, self).__init__()
        self.W_txt = nn.Linear(text_dim, text_dim)

    def forward(self, fea_text, mask=None):
        fea_text = fea_text
        weight_text = self.W_txt(fea_text)  # B*L*C
        if mask is not None:
            mask = mask.permute(0, 2, 1)
            weight_text = weight_text.masked_fill(mask == 0, -1e9)
        weight_text = weight_text.softmax(dim=1)
        weight_text_global_out = weight_text.mean(dim=2)  # B*L
        fea_text_global = fea_text * weight_text
        fea_text_global = fea_text_global.sum(dim=1, keepdim=True)  # B*C*1*1
        return fea_text_global, weight_text_global_out


class Attention(nn.Module):
    def __init__(self, videodim, textdim, attentiondim, groups):
        super(Attention, self).__init__()

        self.groups = groups
        self.q = nn.Linear(textdim, attentiondim)
        self.kv = nn.Linear(videodim, 2*attentiondim)

    def forward(self, videofea, textfea):
        videofea = videofea.permute(0, 2, 1)  # b*t*c
        q = self.q(textfea)  # b*l*c
        kv = self.kv(videofea)
        k, v = kv.chunk(2, dim=-1)
        q = rearrange(q, 'b l (g d) -> b g l d', g=self.groups)
        k = rearrange(k, 'b t (g d) -> b g t d', g=self.groups)
        v = rearrange(v, 'b t (g d) -> b g t d', g=self.groups)
        A = einsum('bgld,bgtd->bglt', [q, k]
                   ).mean(dim=2, keepdim=True)  # b*g*l*t

        att = torch.sigmoid(A)
        out = v.permute(0, 1, 3, 2) * att  # b*g*d*t
        out = rearrange(out, 'b g d t -> b (g d) t')
        return A.mean(dim=[1, 2]), out


class MutanFusion(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers):
        super(MutanFusion, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)

            hv.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.image_transformation_layers = nn.ModuleList(hv)
        #
        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hq.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.ques_transformation_layers = nn.ModuleList(hq)

    def forward(self, ques_emb, img_emb):
        # Pdb().set_trace()
        batch_size = img_emb.size()[0]
        x_mm = []
        for i in range(self.num_layers):
            x_hv = img_emb
            x_hv = self.image_transformation_layers[i](x_hv)

            x_hq = ques_emb
            x_hq = self.ques_transformation_layers[i](x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))

        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, img_emb.shape[1], self.out_dim)
        x_mm = F.tanh(x_mm)
        return x_mm
