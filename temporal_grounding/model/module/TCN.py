from numpy.core.fromnumeric import ptp
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention, Attention2DTAN, AttentionNorm, AttentionCross
from .tanmodule import SparseMaxPool


class TCN(nn.Module):
    def __init__(self, config):
        super(TCN, self).__init__()
        self.config = config
        self.padding_type = config['padding_type']
        self.conv = nn.ModuleList()
        self.dilations = []
        self.attention = nn.ModuleList()
        self.weight_conv = nn.ModuleList()
        self.prenorms = nn.ModuleList()
        for i in range(config['layer_num']):
            dilation = torch.pow(torch.tensor(2), i)
            dilation = int(dilation)
            # dilation = i+1
            self.prenorms.append(nn.LayerNorm(config['attention_dim']))
            self.dilations.append(dilation)
            if config['with_attention']:
                if 'att_norm' in config.keys():
                    self.attention.append(AttentionNorm(config))
                else:
                    self.attention.append(Attention(
                        config['attention_dim'], config['attention_dim'], config['attention_dim'], groups=config['groups']))

            if config['with_mlp']:
                if 'mlp_norm' in config.keys():
                    self.conv.append(
                        nn.Sequential(
                            nn.Linear(config['attention_dim'], config['MLP_dim']),
                            nn.LayerNorm(config['MLP_dim']),
                            nn.Dropout(config['dropout']),
                            nn.ReLU(),
                            nn.Linear(config['MLP_dim'],
                                    config['attention_dim']),
                            nn.LayerNorm(config['attention_dim'])
                        )
                    )
                else:
                    self.conv.append(
                        nn.Sequential(
                            nn.Conv1d(config['attention_dim'], config['MLP_dim'],
                                    3, 1, dilation=dilation, padding=0, bias=False),
                            nn.GroupNorm(4, config['MLP_dim']),
                            nn.Dropout(config['dropout']),
                            nn.ReLU(),
                            nn.Conv1d(config['MLP_dim'],
                                    config['attention_dim'], 1, 1, bias=False),
                            nn.GroupNorm(4, config['attention_dim'])
                        )
                    )

        # self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fea, text_fea, position, mask=None):
        fea = fea + position
        weights = []
        for i in range(len(self.attention)):
            if self.config['prenorm']:
                fea = self.prenorms[i](fea.permute(0, 2, 1)).permute(0, 2, 1)
            res0 = fea
            if self.config['with_attention']:
                weight, fea = self.attention[i](fea, text_fea)
                fea = fea + res0
            res1 = fea
            weights.append(weight)
            if self.config['with_mlp']:
                if 'mlp_norm' not in self.config.keys():
                    if self.padding_type == 'circle':
                        fea = circle_padding(self.dilations[i], fea)
                    elif self.padding_type == 'zero':
                        fea = F.pad(
                            fea, (self.dilations[i], self.dilations[i]), mode='constant', value=0)
                    else:
                        fea = F.pad(
                            fea, (self.dilations[i], self.dilations[i]), mode='replicate')
                    fea = self.conv[i](fea)
                else:
                    fea = self.conv[i](fea.permute(0, 2, 1)).permute(0, 2, 1)
                fea = res1 + fea
        return fea, weights


class TCN0(nn.Module):
    def __init__(self, config):
        super(TCN0, self).__init__()
        self.config = config
        self.padding_type = config['padding_type']
        self.conv = nn.ModuleList()
        self.dilations = []
        self.attention = nn.ModuleList()
        self.weight_conv = nn.ModuleList()
        self.prenorms = nn.ModuleList()
        for i in range(config['layer_num']):
            self.prenorms.append(nn.LayerNorm(config['attention_dim']))
            dilation = torch.pow(torch.tensor(2), i)
            dilation = int(dilation)
            # dilation = i+1
            self.dilations.append(dilation)
            if config['with_attention']:
                self.attention.append(Attention(
                    config['attention_dim'], config['attention_dim'], config['attention_dim'], groups=config['groups']))
                # self.attention.append(AttentionNorm(config))
            else:
                self.attention.append(nn.Identity())
            self.conv.append(
                nn.Sequential(
                    nn.Linear(config['attention_dim'], config['MLP_dim']),
                    nn.LayerNorm(config['MLP_dim']),
                    nn.Dropout(config['dropout']),
                    nn.ReLU(),
                    nn.Linear(config['MLP_dim'],
                              config['attention_dim']),
                    nn.LayerNorm(config['attention_dim'])
                )
            )

        # self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fea, text_fea, text_feal, mask=None):

        weights = []
        for i in range(len(self.attention)):
            fea = self.prenorms[i](fea)
            res0 = fea
            weight, fea = self.attention[i](fea, text_fea)

            fea = fea + res0
            res1 = fea
            weights.append(weight)
            # if self.padding_type == 'circle':
            #     fea = circle_padding(self.dilations[i], fea)
            # elif self.padding_type == 'zero':
            #     fea = F.pad(
            #         fea, (self.dilations[i], self.dilations[i]), mode='constant', value=0)
            # else:
            #     fea = F.pad(
            #         fea, (self.dilations[i], self.dilations[i]), mode='replicate')
            fea = fea.permute(0, 2, 1)
            fea = self.conv[i](fea)
            fea = res1 + fea.permute(0, 2, 1)
        return fea, weights


def circle_padding(padding, feature):
    length_times = feature.shape[-1]
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
        feas.append(feature[:, :, idf])
    feas = torch.stack(feas, dim=2)
    return feas


class TCN2D(nn.Module):
    def __init__(self, config):
        super(TCN2D, self).__init__()
        self.config = config
        self.padding_type = config['padding_type']
        self.conv = nn.ModuleList()
        self.dilations = []
        self.attention = nn.ModuleList()
        self.conv_weight = nn.ModuleList()
        for i in range(config['layer_num']):
            dilation = torch.pow(torch.tensor(2), i)
            dilation = int(dilation)
            self.dilations.append(dilation)
            if config['with_attention']:
                self.attention.append(Attention2DTAN(
                    config['attention_dim'], config['attention_dim'], config['attention_dim'], groups=config['groups']))
            else:
                self.attention.append(nn.Identity())
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(config['attention_dim'], config['MLP_dim'], 3,
                              1, dilation=dilation, padding=dilation, bias=False),
                    nn.GroupNorm(8, config['MLP_dim']),
                    nn.GELU(),
                    nn.Conv2d(config['MLP_dim'],
                              config['attention_dim'], 1, 1),
                    nn.GroupNorm(8, config['attention_dim'])
                )
            )

        # self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fea, text_fea):
        weights = []
        for i in range(len(self.attention)):
            fea0 = fea
            weight, fea = self.attention[i](fea, text_fea)
            weights.append(weight)
            fea = fea + fea0
            fea1 = fea
            fea = self.conv[i](fea)
            fea = fea1 + fea
        return fea, weights


if __name__ == '__main__':
    import json
    feav = torch.randn((4, 256, 100))
    feat = torch.randn((4, 20, 256))
    with open('../../json/config.json') as f:
        config = json.load(f)['model_config']
    tcn = TCN(config)
    fea, weight = tcn(feav, feat)
    print(fea.shape)
    print(len(weight))
