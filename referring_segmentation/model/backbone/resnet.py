import torch
from torch import nn
from model.backbone.frozen_batchnorm import FrozenBatchNorm2d

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, nInputChannels, block, layers, os=32, pretrained=True, model_path=None, norm_layer=FrozenBatchNorm2d):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.norm_layer = norm_layer
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
        elif os == 32:
            strides = [1, 2, 2, 2]
            rates = [1, 1, 1, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])#64 3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])#128 4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])#256 23
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], rate=rates[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model(model_path)


    def _make_layer(self, block, planes, blocks, stride=1, rate=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample, self.norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        layer1_feat = x
        x = self.layer2(x)
        layer2_feat = x
        x = self.layer3(x)
        layer3_feat = x
        x = self.layer4(x)
        return layer1_feat, layer2_feat, layer3_feat, x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, model_path):
        pretrain_dict = torch.load(model_path, map_location='cpu')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class DeeplabResNet(nn.Module):
    def __init__(self, nInputChannels, block, layers, os=32, pretrained=True, model_path=None, norm_layer=FrozenBatchNorm2d):
        super(DeeplabResNet, self).__init__()
        self.backbone = ResNet(nInputChannels, block, layers, os, pretrained=False, norm_layer=norm_layer)

        if pretrained:
            self._load_pretrained_model(model_path)

    def _load_pretrained_model(self, model_path):
        pretrain_dict = torch.load(model_path, map_location='cpu')
        model_dict = {k: v for k,v in pretrain_dict['model_state'].items() if 'backbone' in k}
        self.load_state_dict(model_dict)

    def forward(self, input):
        layer1_feat, layer2_feat, layer3_feat, x = self.backbone(input)
        return layer1_feat, layer2_feat, layer3_feat, x


def ResNet101(nInputChannels=3, os=32, pretrained=False, norm_layer=FrozenBatchNorm2d):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os,
                    pretrained=pretrained, model_path='/media/HardDisk/wwk/pretrained/resnet/resnet101-5d3b4d8f.pth', norm_layer=norm_layer)
    return model

def ResNet18(nInputChannels=3, os=32, pretrained=False, norm_layer=FrozenBatchNorm2d):
    model = ResNet(nInputChannels, BasicBlock, [2, 2, 2, 2], os,
                    pretrained=pretrained, model_path='/media/HardDisk/wwk/pretrained/resnet/resnet18-5c106cde.pth', norm_layer=norm_layer)
    return model

def ResNet34(nInputChannels=3, os=32, pretrained=False, norm_layer=FrozenBatchNorm2d):
    model = ResNet(nInputChannels, BasicBlock, [3, 4, 6, 3], os,
                    pretrained=pretrained, model_path='/media/HardDisk/wwk/pretrained/resnet/resnet34-333f7ec4.pth', norm_layer=norm_layer)
    return model

def ResNet50(nInputChannels=3, os=32, pretrained=False, norm_layer=FrozenBatchNorm2d):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 6, 3], os,
                    pretrained=pretrained, model_path='/media/HardDisk/wwk/pretrained/resnet/resnet50-19c8e357.pth', norm_layer=norm_layer)
    return model


def Deeplab_ResNet50(nInputChannels=3, os=32, pretrained=False, norm_layer=FrozenBatchNorm2d):
    model = DeeplabResNet(nInputChannels, Bottleneck, [3, 4, 6, 3], os, pretrained=pretrained, model_path='./model/pretrained/best_deeplabv3plus_resnet50_voc_os16.pth', norm_layer=norm_layer)
    return model

def Deeplab_ResNet101(nInputChannels=3, os=32, pretrained=False, norm_layer=FrozenBatchNorm2d):
    model = DeeplabResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained, model_path='./model/pretrained/best_deeplabv3plus_resnet101_voc_os16.pth', norm_layer=norm_layer)
    return model


