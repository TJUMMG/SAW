import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.module.attention import GlobalTextPresentation, MuTan
from models.module.RefTransformer import RefTransformer
from torchvision.models._utils import IntermediateLayerGetter
from transformers import RobertaModel, RobertaTokenizerFast

from .criterion import SetCriterion
from .module.decoder import (SpatialDecoder2D, TemporalDecoderAnchor,
                             TemporalDecoderRegression)
from .utils import temporal_separate_to_stack, temporal_stacked_to_separate


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class VideoEncoder(nn.Module):
    def __init__(self, cfg):
        super(VideoEncoder, self).__init__()
        self.cfg = cfg
        backbone = getattr(torchvision.models, cfg.backbone)(
            replace_stride_with_dilation=[False, False, cfg.dilation],
            pretrained=True,
            norm_layer=FrozenBatchNorm2d,
        )
        feature_dims = [256, 512, 1024, 2048]
        if cfg.backbone_multi_scale:
            return_layers = {"layer2": "feat2",
                             "layer3": "feat3", "layer4": "feat4"}
            self.convert = nn.Conv2d(feature_dims[1] + feature_dims[2] + feature_dims[3], cfg.hidden_dim,
                                     1, 1)
        else:
            return_layers = {"layer4": "feat4"}
            self.convert = nn.Conv2d(
                feature_dims[-1], cfg.hidden_dim, 1, 1)
        self.backbone = IntermediateLayerGetter(
            backbone, return_layers=return_layers)

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, frames, frame_masks):
        frames = frames.chunk(5, dim=0)  # split videos to reduce GPU usage
        video_feature = []
        for frame in frames:
            feat_dict = self.backbone(frame)
            feats = [feat_dict['feat4']]
            for k, feat in feat_dict:
                if k != 'feat4':
                    feats.append(F.interpolate(
                        feat, feat_dict['feat4'].shape[2:], mode='bilinear', align_corners=True))
            feats = torch.cat(feats, dim=1)
            video_feature.append(self.convert(feats))
        video_feature = torch.cat(video_feature, dim=0)
        frame_masks = F.interpolate(
            frame_masks[:, None], video_feature.shape[-2:], mode='nearest')
        return video_feature * frame_masks


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        super(TextEncoder, self).__init__()
        self.text_encoder = RobertaModel.from_pretrained(cfg.text_encoder_type)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            cfg.text_encoder_type)
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
        text_memory_resized = self.resizer(text_memory)  # b*l*d
        text_feature = self.global_text(
            text_memory_resized, text_attention_mask)
        return text_feature


class Model(nn.Module):
    def __init__(self, cfg):

        super(Model, self).__init__()
        self.cfg = cfg
        self.video_encoder = VideoEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)

        self.global_attention = MuTan(
            cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.reftrans = RefTransformer(cfg.hidden_dim, cfg.hidden_dim, cfg.dim_feedforward,
                                       cfg.hidden_dim, cfg.layers, dropout=cfg.dropout)

        self.spatial_decoder = SpatialDecoder2D(
            cfg.hidden_dim, cfg.spatial_decoder_hidden_dim, cfg.dilation)

        if cfg.temporal_decoder_type == 'anchor':
            self.temporal_decoder = TemporalDecoderAnchor(
                cfg.hidden_dim, cfg.temporal_window_width, cfg.dropout)  # [current]
        else:
            self.temporal_decoder = TemporalDecoderRegression(
                cfg.hidden_dim, cfg.dropout)

    def forward(self, samples, captions, durations):  # [current]
        frames, frame_masks = samples.tensors, (~samples.mask).float()
        video_feature = self.video_encoder(
            frames, frame_masks)    # [\sigma t_i, c, h, w]
        text_feature = self.text_encoder(captions, frames.device)   # [b, c]

        spatial = generate_spatial_batch(
            video_feature.shape[0], video_feature.shape[-2], video_feature.shape[-1]).to(video_feature.device)
        fused_feature = self.global_attention(
            video_feature, text_feature[..., None, None], spatial)    # [\sigma t_i, c, h, w]

        fused_feature = temporal_stacked_to_separate(
            fused_feature, durations).transpose(1, 2)  # [b, c, t, h, w]

        feature_masks = F.interpolate(
            frame_masks[:, None], fused_feature.shape[-2:], mode='nearest')
        feature_masks = temporal_stacked_to_separate(
            feature_masks, durations).transpose(1, 2)  # [b, 1, t, h, w]

        feature, maps_layers = self.reftrans(
            fused_feature, text_feature, feature_masks, durations)

        feature_stack = temporal_separate_to_stack(feature.transpose(
            1, 2), durations)  # [\sigma t_i, c, h, w] [current]
        spatial_result = self.spatial_decoder(feature_stack)  # [current]
        spatial_map_separate = temporal_stacked_to_separate(
            spatial_result['spatial_map'], durations).transpose(1, 2)  # [b, 1, t, h, w] [current]
        spatial_hm = F.interpolate(
            spatial_map_separate[:, 0], feature.shape[-2:], mode='bilinear', align_corners=True)[:, None]
        feature_obj = ((spatial_hm * feature).sum((-2, -1)) /
                       spatial_hm.sum((-2, -1)))  # [b, c, t]

        feature_global = feature.mean((-2, -1))  # [b, c, t]
        temporal_result = self.temporal_decoder(feature_global, feature_obj)

        result_dict = {}
        result_dict.update({
            'maps': maps_layers
        })
        result_dict.update(spatial_result)
        result_dict.update(temporal_result)
        return result_dict


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
    spatial_batch_val = np.zeros(
        (N, featmap_H, featmap_W, 8), dtype=np.float32)
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
        'spatial_map_loss': cfg.loss_spatial_map
    }
    if cfg.temporal_decoder_type == 'anchor':
        weight_dict.update(
            {
                "temporal_cls_loss": cfg.loss_temporal_cls,
                "temporal_align_loss": cfg.loss_temporal_align,
            }
        )
    elif cfg.temporal_decoder_type == 'regression':
        weight_dict.update(
            {
                "temporal_score_loss": cfg.loss_temporal_score,
                "temporal_reg_loss": cfg.loss_temporal_reg,
                "temporal_iou_loss": cfg.loss_temporal_iou,
            }
        )
    return model, criterion, weight_dict
