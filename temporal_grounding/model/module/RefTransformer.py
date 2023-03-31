import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention


class RefTransformer(nn.Module):
    def __init__(self, config):
        super(RefTransformer, self).__init__()
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
                self.attention.append(Attention(
                    config['attention_dim'], config['attention_dim'], config['attention_dim'], groups=config['groups']))

            if config['with_mlp']:
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
                if self.padding_type == 'circle':
                    fea = circle_padding(self.dilations[i], fea)
                elif self.padding_type == 'zero':
                    fea = F.pad(
                        fea, (self.dilations[i], self.dilations[i]), mode='constant', value=0)
                else:
                    fea = F.pad(
                        fea, (self.dilations[i], self.dilations[i]), mode='replicate')
                fea = self.conv[i](fea)
                fea = res1 + fea
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
