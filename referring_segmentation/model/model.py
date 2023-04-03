from model.backbone.resnet import ResNet50, ResNet34, ResNet101, ResNet18, Deeplab_ResNet50, Deeplab_ResNet101
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.attention import GlobalTextPresentation, GlobalAttention, MuTan
from model.module.TCN import TCN
import numpy as np
from model.backbone.frozen_batchnorm import FrozenBatchNorm2d
from model.module.aspp import ASPP


class Decoder(nn.Module):
    def __init__(self, config, low_level_dim, out_feature_dim, norm_type='GroupNorm'):
        super(Decoder, self).__init__()
        self.aspp = ASPP(config['TCN_feature_dim'], config['video_feature_dim'], [1, 6, 12, 18], norm_type)

        if config['norm_type']=='GroupNorm':
            self.convert = nn.Sequential(
                nn.Conv2d(low_level_dim, 48, 3, 1, 1, bias=False),
                nn.GroupNorm(8, 48),
                nn.ReLU(inplace=True)
            )
            self.tail = nn.Sequential(
                nn.Conv2d(48 + out_feature_dim, 256, 3, 1, 1, bias=False),
                nn.GroupNorm(8, 256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, 1, 1)
            )
        else:
            self.convert = nn.Sequential(
                nn.Conv2d(low_level_dim, 48, 3, 1, 1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            )
            self.tail = nn.Sequential(
                nn.Conv2d(48 + out_feature_dim, 256, 3, 1, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, 1, 1)
            )
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, low_level_feat, out_fea):
        out_fea = self.aspp(out_fea)
        low_level_feat = self.convert(low_level_feat)
        out_fea = F.interpolate(out_fea, low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        fea = torch.cat([out_fea, low_level_feat], dim=1)
        out = self.tail(fea)
        return out

class VideoEncoder(nn.Module):
    def __init__(self, config):
        super(VideoEncoder, self).__init__()
        self.config = config
        if config['frozen_batchnorm']:
            norm_layer = FrozenBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        if config['backbone'] == 'resnet50':
            self.backbone = ResNet50(config['input_dim'], config['os'], pretrained=True, norm_layer=norm_layer)
            self.feature_dims = [256, 512, 1024, 2048]
        elif config['backbone'] == 'resnet101':
            self.backbone = ResNet101(config['input_dim'], config['os'], pretrained=True, norm_layer=norm_layer)
            self.feature_dims = [256, 512, 1024, 2048]
        elif config['backbone'] == 'resnet34':
            self.backbone = ResNet34(config['input_dim'], config['os'], pretrained=True, norm_layer=norm_layer)
            self.feature_dims = [64, 128, 256, 512]
        elif config['backbone'] == 'resnet18':
            self.backbone = ResNet18(config['input_dim'], config['os'], pretrained=True, norm_layer=norm_layer)
            self.feature_dims = [64, 128, 256, 512]
        elif config['backbone'] == 'deeplab_resnet50':
            self.backbone = Deeplab_ResNet50(config['input_dim'], 16, pretrained=True, norm_layer=norm_layer)
            self.feature_dims = [256, 512, 1024, 2048]
        elif config['backbone'] == 'deeplab_resnet101':
            self.backbone = Deeplab_ResNet101(config['input_dim'], 16, pretrained=True, norm_layer=norm_layer)
            self.feature_dims = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError
        if config['backbone_multi_scale']:
            self.convert = nn.Conv2d(self.feature_dims[1] + self.feature_dims[2] + self.feature_dims[3], config['video_feature_dim'],
                                     1, 1)
        else:
            self.convert = nn.Conv2d(self.feature_dims[3], config['video_feature_dim'], 1, 1)

    def forward(self, frame):
        layer1_feat, layer2_feat, layer3_feat, layer4_feat = self.backbone(frame)
        if self.config['backbone_multi_scale']:
            layer2_feat = F.interpolate(layer2_feat, layer4_feat.shape[2:], mode='bilinear', align_corners=True)
            layer3_feat = F.interpolate(layer3_feat, layer4_feat.shape[2:], mode='bilinear', align_corners=True)
            fea_cat = torch.cat([layer4_feat, layer3_feat, layer2_feat], dim=1)
            fea_f = self.convert(fea_cat)
        else:
            fea_f = self.convert(layer4_feat)
        return  fea_f, layer1_feat

class TextEncoder(nn.Module):
    def __init__(self, config):
        self.config = config
        super(TextEncoder, self).__init__()
        if config['gru_bidirection']:
            self.backbone = nn.GRU(config['embedding_dim'], config['text_feature_dim'], batch_first=True, bidirectional=True)
        else:
            self.backbone = nn.GRU(config['embedding_dim'], config['text_feature_dim'], batch_first=True, bidirectional=False)

    def forward(self, text, embedding_length):
        text = torch.nn.utils.rnn.pack_padded_sequence(text, list(embedding_length), batch_first=True, enforce_sorted=False)
        word_embedding, _ = self.backbone(text)
        word_embedding, _ = torch.nn.utils.rnn.pad_packed_sequence(word_embedding, True)
        if self.config['gru_bidirection']:
            word_embedding = word_embedding.view(word_embedding.shape[0], word_embedding.shape[1], 2, -1)
            word_embedding = torch.mean(word_embedding, dim=2)
        word_embedding = word_embedding.permute(0, 2, 1)
        return  word_embedding

class GlobalLocalTCN(nn.Module):
    def __init__(self, config):
        super(GlobalLocalTCN, self).__init__()
        self.config = config
        self.global_text = GlobalTextPresentation(config['text_feature_dim'])
        if config['is_global_attention']:
            if config['global_fuse_type'] == 'attention':
                self.global_attention = GlobalAttention(config['video_feature_dim'], config['text_feature_dim'],
                                                        config['attention_dim'])
                self.convert_global = nn.Conv2d(config['video_feature_dim'] + config['text_feature_dim'] + 8,
                                                config['attention_dim'], 1, 1)
            else:
                self.global_attention = MuTan(config['video_feature_dim'], config['text_feature_dim'],
                                              config['attention_dim'])
        else:
            self.convert_global = nn.Conv2d(config['video_feature_dim'] + config['text_feature_dim'] + 8,
                                            config['attention_dim'], 1, 1)

        self.projection = nn.Conv3d(config['attention_dim'], config['TCN_feature_dim'], 1, 1)
        self.TCN = TCN(config['text_feature_dim'], config['TCN_feature_dim'], config['TCN_hidden_dim'],
                       config['TCN_feature_dim'], config['layer_num'], config['padding_type'],
                       config['is_local_attention'], config['conv_type'], config['local_fuse_type'],
                       groups=config['groups'], norm_type=config['norm_type'])

    def forward(self, videofeas, word_embedding, embedding_length):
        n_frames = len(videofeas)
        embedding_mask = torch.zeros((word_embedding.shape[0], 1, word_embedding.shape[-1])).to(
            word_embedding.device)
        for b in range(embedding_mask.shape[0]):
            embedding_mask[b, :, :int(embedding_length[b])] = 1
        mask_global = embedding_mask
        global_text = self.global_text(word_embedding, mask_global)  # B*C2*1*1
        feas = []
        for videofea in videofeas:
            spatial = generate_spatial_batch(videofea.shape[0], videofea.shape[-2], videofea.shape[-1])
            spatial = torch.from_numpy(spatial).permute(0, 3, 1, 2).to(videofea.device)  # B*8*H*W
            if self.config['is_global_attention']:
                if self.config['global_fuse_type'] == 'attention':
                    global_text_tiled = F.interpolate(global_text, size=videofea.size()[2:], mode='bilinear',
                                                      align_corners=True)
                    global_fea = torch.cat([videofea, global_text_tiled, spatial], dim=1)  # B*(C1+C2+8)*H*W
                    res = self.convert_global(global_fea)
                    global_fea = self.global_attention(global_fea, global_text)  # B*C3*H*W
                    global_fea = global_fea + res
                else:
                    global_fea = self.global_attention(videofea, global_text, spatial)
            else:
                global_text_tiled = F.interpolate(global_text, size=videofea.size()[2:], mode='bilinear',
                                                  align_corners=True)
                global_fea = torch.cat([videofea, global_text_tiled, spatial], dim=1)  # B*(C1+C2+8)*H*W
                global_fea = self.convert_global(global_fea)
            feas.append(global_fea)
        fea = torch.stack(feas, dim=2)
        frame_mask = torch.ones((fea.shape[0], fea.shape[3] * fea.shape[4], 1)).to(word_embedding.device)
        mask_local = torch.matmul(frame_mask, embedding_mask)
        fea = self.projection(fea)
        if self.config['filter_type'] != 'global':
            fea, maps = self.TCN(fea, word_embedding, mask_local)
        else:
            fea, maps = self.TCN(fea, global_text.squeeze(-1), mask_local)
        feas = fea.chunk(n_frames, dim=2)
        return feas, maps

class Model(nn.Module):
    def __init__(self, config):

        super(Model, self).__init__()
        self.config = config
        self.video_encoder = VideoEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.bone = GlobalLocalTCN(config)
        self.decoder = Decoder(config, self.video_encoder.feature_dims[0], config['TCN_feature_dim'], config['norm_type'])

        if config['frozen_backbone']:
            for i in self.video_encoder.backbone.parameters():
                i.requires_grad = False

    def forward(self, frames, text, embedding_length):
        text_fea = self.text_encoder(text, embedding_length)
        video_feas = []
        low_level_feas = []
        for frame in frames:
            video_fea, low_level_fea = self.video_encoder(frame)
            video_feas.append(video_fea)
            low_level_feas.append(low_level_fea)
        out_feas, maps = self.bone(video_feas, text_fea, embedding_length)
        outs = []
        for i in range(len(frames)):
            fea = out_feas[i].squeeze(2)
            out = self.decoder(low_level_feas[i], fea)
            out = F.interpolate(out, frames[i].shape[2:], mode='bilinear', align_corners=True)
            outs.append(out)

        return outs, maps

def generate_spatial_batch(N, featmap_H, featmap_W):
    spatial_batch_val = np.zeros((N, featmap_H, featmap_W, 8), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w+1) / featmap_W * 2 - 1
            xctr = (xmin+xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h+1) / featmap_H * 2 - 1
            yctr = (ymin+ymax) / 2
            spatial_batch_val[:, h, w, :] = \
                [xmin, ymin, xmax, ymax, xctr, yctr, 1/featmap_W, 1/featmap_H]
    return spatial_batch_val
