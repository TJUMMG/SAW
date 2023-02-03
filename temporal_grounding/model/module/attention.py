from utils.utils import generate_candidates
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
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
        # self.k = nn.Linear(videodim, attentiondim)
        self.kv = nn.Linear(videodim, 2*attentiondim)

    def forward(self, videofea, textfea):
        videofea = videofea.permute(0, 2, 1)  # b*t*c
        q = self.q(textfea)  # b*l*c
        # k = self.k(videofea)
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


class AttentionCross(nn.Module):
    def __init__(self, videodim, textdim, attentiondim, groups):
        super(AttentionCross, self).__init__()

        self.groups = groups
        self.q = nn.Linear(textdim, attentiondim)
        self.k = nn.Linear(videodim, attentiondim)
        self.scale = sqrt(attentiondim)
        self.vv = nn.Linear(textdim, attentiondim)
        self.vt = nn.Linear(videodim, attentiondim)

    def forward(self, videofea, textfea, mask=None):
        videofea = videofea.permute(0, 2, 1)  # b*t*c
        q = self.q(textfea)  # b*l*c
        k = self.k(videofea)

        vv = self.vv(textfea)
        vt = self.vt(videofea)

        q = rearrange(q, 'b l (g d) -> b g l d', g=self.groups)
        k = rearrange(k, 'b t (g d) -> b g t d', g=self.groups)
        att = einsum('bgld,bgtd->bglt', [q, k])  # b*g*l*t
        if mask != None:
            mask = mask.permute(0, 2, 1).unsqueeze(1)
            att = att.masked_fill(mask == 0, -1e9)
        att_t = att.softmax(dim=-1) / self.scale  # b*g*l*t
        att_v = att.permute(0, 1, 3, 2).softmax(dim=-1) / self.scale  # b*g*t*l

        vv = rearrange(vv, 'b t (g d) -> b g t d', g=self.groups)  # b*g*l*d
        vt = rearrange(vt, 'b t (g d) -> b g t d', g=self.groups)  # b*g*t*d

        out_v = einsum('bgtl,bgld->bgtd', [att_v, vv])
        out_t = einsum('bglt,bgtd->bgld', [att_t, vt])

        out_v = rearrange(out_v, 'b g t d -> b (g d) t', g=self.groups)
        out_t = rearrange(out_t, 'b g l d -> b l (g d)', g=self.groups)

        return out_v, out_t, None


class Attention2DTAN(nn.Module):
    def __init__(self, videodim, textdim, attentiondim, groups):
        super(Attention2DTAN, self).__init__()

        self.groups = groups
        self.q = nn.Linear(textdim, attentiondim)
        self.k = nn.Conv2d(videodim, attentiondim, 1, 1)
        self.v = nn.Conv2d(videodim, attentiondim, 1, 1)

    def forward(self, videofea, textfea):
        q = self.q(textfea)  # b*l*c
        k = self.k(videofea)  # b*c*t*t
        v = self.v(videofea)  # b*c*t*t

        q = rearrange(q, 'b l (g d) -> b g d l', g=self.groups)
        k = rearrange(k, 'b (g d) h w -> b g d h w', g=self.groups)
        v = rearrange(v, 'b (g d) h w -> b g d h w', g=self.groups)

        q = q.unsqueeze(-1)  # b*g*d*1*1
        A = (q * k).sum(dim=2, keepdim=True)  # b*g*1*t*t
        att = torch.sigmoid(A)
        out = v * att  # b*g*d*t
        out = rearrange(out, 'b g d h w -> b (g d) h w')
        return A.mean(dim=1), out


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


class AttentionWordWise(nn.Module):
    def __init__(self, videodim, textdim, attentiondim, groups):
        super(AttentionWordWise, self).__init__()
        self.W_w = nn.Linear(2 * textdim, attentiondim)
        self.W_g = nn.Linear(textdim, attentiondim)
        self.W_l = nn.Linear(textdim, attentiondim)
        self.W_weight = nn.Linear(attentiondim, 1)

        self.W_v1 = nn.Linear(videodim, attentiondim)
        self.W_v2 = nn.Linear(videodim, attentiondim)
        self.W_v3 = nn.Linear(videodim, attentiondim)

    def forward(self, videofea, textfea_g, textfea_l, mask=None):
        videofea = videofea.permute(0, 2, 1)
        h = self.W_w(torch.cat([textfea_l, textfea_g.repeat(
            1, textfea_l.shape[1], 1)], dim=-1)).sigmoid()

        fea_t = h * self.W_g(textfea_g) + (1 - h) * \
            self.W_l(textfea_l)  # b*l*d

        fea_v1 = self.W_v1(videofea)  # b*t*d
        fea_v2 = self.W_v2(videofea)  # b*t*d
        fea_v3 = self.W_v3(videofea)

        att = torch.matmul(fea_t, fea_v1.permute(0, 2, 1))  # b*l*t
        att = att / (fea_v1.shape[-1] ** (0.5))
        if mask != None:
            mask = mask.permute(0, 2, 1)
            att = att.masked_fill(mask == 0, -1e9).softmax(dim=-1)
        weight_t = torch.matmul(att, fea_v2)  # b*l*d
        weight_t = self.W_weight(weight_t)  # b*1*t

        att_v = (att * weight_t).sum(dim=1).unsqueeze(-1)
        out_fea = att_v.sigmoid() * fea_v3
        return att_v, out_fea.permute(0, 2, 1)


class AttentionNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config['attention_dim'], 4, config['dropout'])

    def forward(self, feav, feat):

        fea_out, weight = self.attention(query=feav.permute(
            2, 0, 1), key=feat.permute(1, 0, 2), value=feat.permute(1, 0, 2))
        return weight.squeeze(), fea_out.permute(1, 2, 0)


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.prenorm = nn.LayerNorm(config['attention_dim'])

        self.attention_self = nn.MultiheadAttention(
            config['attention_dim'], 4, config['dropout'])
        self.norm0 = nn.LayerNorm(config['attention_dim'])
        self.drop0 = nn.Dropout(config['dropout'])
        self.attention = nn.MultiheadAttention(
            config['attention_dim'], 4, config['dropout'])

        self.mlp = nn.Sequential(
            nn.Linear(config['attention_dim'], config['attention_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['attention_dim'], config['attention_dim'])
        )
        self.norm1 = nn.LayerNorm(config['attention_dim'])
        self.norm2 = nn.LayerNorm(config['attention_dim'])
        self.drop1 = nn.Dropout(config['dropout'])
        self.drop2 = nn.Dropout(config['dropout'])

    def forward(self, anchor_fea, fea):
        """
        anchor_fea: b*l*d
        fea: b*n*d
        """
        anchor_fea = self.prenorm(anchor_fea)
        anchor_fea1 = self.attention_self(anchor_fea.permute(1, 0, 2), anchor_fea.permute(1, 0, 2), anchor_fea.permute(1, 0, 2))[0]
        anchor_fea1 = anchor_fea1.permute(1, 0, 2)
        anchor_fea = anchor_fea + self.drop0(anchor_fea1)
        anchor_fea = self.norm0(anchor_fea)
        fea_out, _ = self.attention(query=anchor_fea.permute(1, 0, 2), key=fea.permute(1, 0, 2), value=fea.permute(1, 0, 2))
        fea_out = fea_out.permute(1, 0, 2)
        fea_out = anchor_fea + self.drop1(fea_out)
        fea_out = self.norm1(fea_out)
        fea_out1 = self.mlp(fea_out)
        fea_out = fea_out + self.drop2(fea_out1)

        return fea_out


if __name__ == '__main__':
    feav = torch.randn((4, 256, 100))
    feat = torch.randn((4, 20, 256))
    att = Attention(256, 256, 256, 8)
    A, out = att(feav, feat)
    print(A.shape)
    print(out.shape)