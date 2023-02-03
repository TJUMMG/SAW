import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate, norm_type='GroupNorm'):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        if norm_type=='GroupNorm':
            self.bn = nn.GroupNorm(8, planes)
        else:
            self.bn = nn.BatchNorm2d(planes)
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=rate, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, inplanes, planes, rates, norm_type='GroupNorm'):
        super(ASPP, self).__init__()

        self.aspp1 = ASPP_module(inplanes, planes, rate=rates[0], norm_type=norm_type)
        self.aspp2 = ASPP_module(inplanes, planes, rate=rates[1], norm_type=norm_type)
        self.aspp3 = ASPP_module(inplanes, planes, rate=rates[2], norm_type=norm_type)
        self.aspp4 = ASPP_module(inplanes, planes, rate=rates[3], norm_type=norm_type)

        self.relu = nn.ReLU()

        if norm_type=='GroupNorm':
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
                nn.GroupNorm(8, planes),
                nn.ReLU()
            )
            self.bn1 = nn.GroupNorm(8, planes)
        else:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU()
            )
            self.bn1 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(planes*5, planes, 1, bias=False)
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x