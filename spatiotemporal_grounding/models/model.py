from models.backbone.resnet import ResNet50, ResNet101, Deeplab_ResNet50, Deeplab_ResNet101
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module.attention import GlobalTextPresentation, MuTan
from models.module.TCN import TCN
import numpy as np
from models.backbone.frozen_batchnorm import FrozenBatchNorm2d
from transformers import RobertaModel, RobertaTokenizerFast
from .criterion import SetCriterion

class TemporalDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.reg_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim*2),
            nn.Linear(cfg.hidden_dim*2,
                      cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, len(cfg.temporal_window_width) * 2)
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim*2),
            nn.Linear(cfg.hidden_dim*2, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, len(cfg.temporal_window_width))
        )
    def forward(self, feature, spatial_hm):
        spatial_hm = F.interpolate(spatial_hm, feature.shape[-2:], mode='bilinear', align_corners=True)
        feature_local = ((spatial_hm.sigmoid() * feature).sum((-2,-1)) / spatial_hm.sum((-2, -1))).unsqueeze(0)
        feature_global = feature.mean((-2, -1)).unsqueeze(0)    # b*t*c
        feature_flatten = torch.cat([feature_local, feature_global], dim=-1)
        offset = self.reg_head(feature_flatten)
        offset = offset.contiguous().view(-1, offset.shape[1] * len(self.cfg.temporal_window_width), 2) # b*(t*n_box)*2
        pred_score = self.cls_head(feature_flatten)
        pred_score = torch.sigmoid(pred_score).contiguous().view(pred_score.size(0), -1)  # b*(t*n_box)
        return offset, pred_score

class SpatialDecoder(nn.Module):
    def __init__(self, cfg):
        super(SpatialDecoder, self).__init__()
        scale = 4 if cfg.dilation else 8
        if scale == 8:
            hidden_dims = [64, 64, 64]
        elif scale == 4:
            hidden_dims = [64, 64]
        
        up = []
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                up.append(nn.ConvTranspose2d(cfg.hidden_dim, cfg.spatial_decoder_hidden_dim, 3, 2, 1, 1, bias=False))
            else:
                up.append(
                    nn.ConvTranspose2d(cfg.spatial_decoder_hidden_dim, cfg.spatial_decoder_hidden_dim, 3, 2, 1, 1, bias=False))
            up.append(nn.GroupNorm(4, cfg.spatial_decoder_hidden_dim))
            up.append(nn.ReLU())
        self.up = nn.Sequential(*up)
        self.hm = nn.Sequential(
            nn.Conv2d(cfg.spatial_decoder_hidden_dim, cfg.spatial_decoder_hidden_dim, 1),
            nn.GroupNorm(4, cfg.spatial_decoder_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(cfg.spatial_decoder_hidden_dim, 1, 1)
        )
        self.reg = nn.Sequential(
            nn.Conv2d(cfg.spatial_decoder_hidden_dim, cfg.spatial_decoder_hidden_dim, 1),
            nn.GroupNorm(4, cfg.spatial_decoder_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(cfg.spatial_decoder_hidden_dim, 2, 1)
        )
        
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feature):
        feature = self.up(feature)
        heatmap = self.hm(feature)
        regression = self.reg(feature)
        return heatmap, regression

class VideoEncoder(nn.Module):
    def __init__(self, cfg):
        super(VideoEncoder, self).__init__()
        self.cfg = cfg
        norm_layer = FrozenBatchNorm2d
        os = 32 if not cfg.dilation else 16

        if cfg.backbone == 'resnet50':
            self.backbone = ResNet50(3, os, pretrained=True, norm_layer=norm_layer)
            self.feature_dims = [256, 512, 1024, 2048]
        elif cfg.backbone == 'resnet101':
            self.backbone = ResNet101(3, os, pretrained=True, norm_layer=norm_layer)
            self.feature_dims = [256, 512, 1024, 2048]
        elif cfg.backbone == 'deeplab_resnet50':
            self.backbone = Deeplab_ResNet50(3, os, pretrained=True, norm_layer=norm_layer)
            self.feature_dims = [256, 512, 1024, 2048]
        elif cfg.backbone == 'deeplab_resnet101':
            self.backbone = Deeplab_ResNet101(3, os, pretrained=True, norm_layer=norm_layer)
            self.feature_dims = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        if cfg.backbone_multi_scale:
            self.convert = nn.Conv2d(self.feature_dims[1] + self.feature_dims[2] + self.feature_dims[3], cfg.hidden_dim,
                                     1, 1)
        else:
            self.convert = nn.Conv2d(self.feature_dims[3], cfg.hidden_dim, 1, 1)

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                    p.requires_grad_(False)

    def forward(self, frames, frame_masks):
        frames = frames.chunk(5, dim=0)
        video_feature = []
        for frame in frames:
            _, layer2_feat, layer3_feat, layer4_feat = self.backbone(frame)
            if self.cfg.backbone_multi_scale:
                layer2_feat = F.interpolate(layer2_feat, layer4_feat.shape[2:], mode='bilinear', align_corners=True)
                layer3_feat = F.interpolate(layer3_feat, layer4_feat.shape[2:], mode='bilinear', align_corners=True)
                fea_cat = torch.cat([layer4_feat, layer3_feat, layer2_feat], dim=1)
                fea_f = self.convert(fea_cat)
            else:
                fea_f = self.convert(layer4_feat)
            video_feature.append(fea_f)
        video_feature = torch.cat(video_feature, dim=0)
        frame_masks = F.interpolate(frame_masks[:, None], video_feature.shape[-2:], mode='nearest')
        return  video_feature * frame_masks

class TextEncoder(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        super(TextEncoder, self).__init__()
        self.text_encoder = RobertaModel.from_pretrained(cfg.text_encoder_type)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(cfg.text_encoder_type)
        if cfg.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                    p.requires_grad_(False)

        self.resizer = FeatureResizer(
            input_feat_size=self.text_encoder.config.hidden_size,
            output_feat_size=cfg.hidden_dim,
            dropout=0.1,
        )
        self.global_text = GlobalTextPresentation(cfg.hidden_dim)

    def forward(self, text, device):
        tokenized = self.tokenizer.batch_encode_plus(
                    text, padding="longest", return_tensors="pt"
                ).to(device)
        encoded_text = self.text_encoder(**tokenized)

        text_memory = encoded_text.last_hidden_state
        text_attention_mask = tokenized.attention_mask.unsqueeze(-1)
        text_memory_resized = self.resizer(text_memory) # b*l*d
        text_feature = self.global_text(text_memory_resized, text_attention_mask)
        return  text_feature

class Model(nn.Module):
    def __init__(self, cfg):

        super(Model, self).__init__()
        self.cfg = cfg
        self.video_encoder = VideoEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)

        self.global_attention = MuTan(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.reftrans = TCN(cfg.hidden_dim, cfg.hidden_dim, cfg.dim_feedforward, cfg.hidden_dim, cfg.layers)

        self.spatial_decoder = SpatialDecoder(cfg)
        self.temporal_decoder = TemporalDecoder(cfg)


    def forward(self, samples, captions):
        frames, frame_masks = samples.tensors, (~samples.mask).float()
        video_feature = self.video_encoder(frames, frame_masks)    # t*c*h*w
        text_feature = self.text_encoder(captions, frames.device)   # b*c

        spatial = generate_spatial_batch(video_feature.shape[0], video_feature.shape[-2], video_feature.shape[-1]).to(video_feature.device)
        fused_feature = self.global_attention(video_feature, text_feature[..., None, None], spatial)
        fused_feature = fused_feature.unsqueeze(0).transpose(1, 2) # b*c*t*h*w
        feature_masks = F.interpolate(frame_masks[:, None], fused_feature.shape[-2:], mode='nearest')
        feature_masks = feature_masks.transpose(0, 1).unsqueeze(1)   # b*t*h*w

        feature, maps_layers = self.reftrans(fused_feature, text_feature, feature_masks)
        feature = feature.squeeze(0).transpose(0, 1)    # t*c*h*w

        spatial_map, spatial_wh = self.spatial_decoder(feature)
        temporal_offset, temporal_score = self.temporal_decoder(feature, spatial_map)
        
        return {
            'spatial_map': spatial_map,
            'spatial_wh': spatial_wh,
            'temporal_offset': temporal_offset,
            'temporal_score': temporal_score,
            'maps': maps_layers
        }

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

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
    return torch.from_numpy(spatial_batch_val).permute(0, 3, 1, 2)

def build_model(cfg):
    model = Model(cfg)
    criterion = SetCriterion(cfg)
    weight_dict = {
        "spatial_hm_loss": cfg.loss_spatial_hm,
        "spatial_wh_loss": cfg.loss_spatial_wh,
        "temporal_cls_loss": cfg.loss_temporal_cls,
        "temporal_reg_loss": cfg.loss_temporal_reg,
        'spatial_map_loss': cfg.loss_spatial_map
    }
    return model, criterion, weight_dict