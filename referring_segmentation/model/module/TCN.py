import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.attention import LocalAttention, RelevanceFilter


class TCN(nn.Module):
    def __init__(self, text_dim, inchannel, hidden_channel, outchannel, layers=8, padding_type='zero', with_local_attention=True, conv_type='3D', local_attention_type='relevance_filter', groups=8, norm_type='GroupNorm'):
        super(TCN, self).__init__()
        self.padding_type = padding_type
        self.with_local_attention = with_local_attention
        self.local_attention_type = local_attention_type
        self.conv_time = nn.ModuleList()
        self.conv_spatial = nn.ModuleList()
        self.conv_convert = nn.ModuleList()
        self.dilations = []
        self.local_attention = nn.ModuleList()
        # self.global_txt_W = nn.ModuleList()
        for i in range(layers):
            # self.global_txt_W.append(nn.Linear(text_dim, hidden_channel))
            dilation = torch.pow(torch.tensor(2), i)
            dilation = int(dilation)
            self.dilations.append(dilation)
            if with_local_attention:
                if local_attention_type == 'attention':
                    self.local_attention.append(LocalAttention(inchannel, text_dim, inchannel))
                else:
                    self.local_attention.append(RelevanceFilter(text_dim, inchannel, inchannel, groups=groups))
            else:
                self.local_attention.append(nn.Identity())

            if conv_type == '3D':
                self.conv_spatial.append(nn.Identity())
                if norm_type == "GroupNorm":
                    self.conv_time.append(
                        nn.Sequential(
                            nn.Conv3d(inchannel, hidden_channel, (3, 3, 3), 1, (0, 1, 1), (dilation, 1, 1), bias=False),
                            nn.GroupNorm(8, hidden_channel),
                            nn.ReLU(inplace=True))
                        )
                else:
                    self.conv_time.append(
                        nn.Sequential(
                            nn.Conv3d(inchannel, hidden_channel, (3, 3, 3), 1, (0, 1, 1), (dilation, 1, 1), bias=False),
                            nn.BatchNorm3d(hidden_channel),
                            nn.ReLU(inplace=True))
                        )

            else:
                if norm_type == "GroupNorm":
                    self.conv_spatial.append(
                        nn.Sequential(
                            nn.Conv3d(inchannel, hidden_channel, (1, 3, 3), 1, (0, 1, 1), (1, 1, 1), bias=False),
                            nn.GroupNorm(8, hidden_channel),
                            nn.ReLU(inplace=True)
                        )
                    )
                    self.conv_time.append(
                        nn.Sequential(
                            nn.Conv3d(hidden_channel, hidden_channel, (3, 1, 1), (1, 1, 1), (0, 0, 0), (dilation, 1, 1), bias=False),
                            nn.GroupNorm(8, hidden_channel),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    self.conv_spatial.append(
                        nn.Sequential(
                            nn.Conv3d(inchannel, hidden_channel, (1, 3, 3), 1, (0, 1, 1), (1, 1, 1), bias=False),
                            nn.BatchNorm3d(hidden_channel),
                            nn.ReLU(inplace=True)
                        )
                    )
                    self.conv_time.append(
                        nn.Sequential(
                            nn.Conv3d(hidden_channel, hidden_channel, (3, 1, 1), (1, 1, 1), (0, 0, 0), (dilation, 1, 1), bias=False),
                            nn.BatchNorm3d(hidden_channel),
                            nn.ReLU(inplace=True)
                        )
                    )
            if norm_type == "GroupNorm":
                self.conv_convert.append(
                    nn.Sequential(
                        nn.Conv3d(hidden_channel, outchannel, 1, 1, bias=False),
                        nn.GroupNorm(8, outchannel)
                    )
                )
            else:
                self.conv_convert.append(
                    nn.Sequential(
                        nn.Conv3d(hidden_channel, outchannel, 1, 1, bias=False),
                        nn.BatchNorm3d(outchannel)
                    )
                )
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fea, fea_text, mask_local):
        fea_text = fea_text.permute(0, 2, 1)  # B*L*C
        maps_layers = []
        maps_sep_layers = []
        for i in range(len(self.conv_time)):
            res0 = fea

            if self.with_local_attention:
                if self.local_attention_type == 'attention':
                    fea = self.local_attention[i](fea, fea_text, mask_local)
                else:
                    maps, fea, maps_sep = self.local_attention[i](fea, fea_text)
                    maps_layers.append(maps)
                    maps_sep_layers.append(maps_sep)
                fea = res0 + fea

            res1 = fea
            fea = self.conv_spatial[i](fea)

            if self.padding_type == 'circle':
                fea = circle_padding(self.dilations[i], fea)
            elif self.padding_type == 'zero':
                fea = F.pad(fea, (0, 0, 0, 0, self.dilations[i], self.dilations[i]), mode='constant', value=0)
            else:
                fea = F.pad(fea, (0, 0, 0, 0, self.dilations[i], self.dilations[i]), mode='circular')

            fea = self.conv_time[i](fea)  # B*C*T
            fea = self.conv_convert[i](fea)
            fea = fea + res1
        return fea, maps_layers, maps_sep_layers


def circle_padding(padding, feature):
    length_times = feature.shape[2]
    index = list(range(0, length_times)) + list(range(length_times - 2, 0, -1))
    total_num = 2 * padding + length_times
    num_c = padding // len(index)
    if num_c * len(index) < padding:
        num_c = num_c + 1
    expand_number = num_c * len(index) - padding
    index_f = []
    for n in range(num_c):
        index = index + index + index
    for i in range(expand_number, expand_number + total_num):
        index_f.append(index[i])

    feas = []
    for idf in index_f:
        feas.append(feature[:, :, idf, :, :].unsqueeze(2))
    feas = torch.cat(feas, dim=2)
    return feas
