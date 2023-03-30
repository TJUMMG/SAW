from typing import Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialDecoderBlock(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 64, scale_layer_num: int = 3,
                 out_hm_dim: int = 1, out_reg_dim: int = 1, phase: str = '1D'):
        super().__init__()
        self.phase = phase
        if phase == '2D':
            conv = nn.Conv2d
            upsample_mode = 'bilinear'
        elif phase == '1D':
            conv = nn.Conv1d
            upsample_mode = 'linear'
        else:
            assert NotImplementedError

        up = []
        for i, _ in enumerate(range(scale_layer_num)):
            if i == 0:
                up.append(conv(in_dim, hidden_dim, 3, 1, 1, bias=False),)
            else:
                up.append(conv(hidden_dim, hidden_dim, 3, 1, 1, bias=False))
            up.append(nn.GroupNorm(4, hidden_dim))
            up.append(nn.ReLU())
            up.append(nn.Upsample(scale_factor=2,
                      mode=upsample_mode, align_corners=True))
        self.up = nn.Sequential(*up)
        self.hm = nn.Sequential(
            conv(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(),
            conv(hidden_dim, out_hm_dim, 1)
        )
        self.reg = nn.Sequential(
            conv(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(),
            conv(hidden_dim, out_reg_dim, 1)
        )
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feature: Tensor) -> Tuple[Tensor, Tensor]:
        feature = self.up(feature)
        heatmap = self.hm(feature)
        regression = self.reg(feature)
        return heatmap, regression


class SpatialDecoder2D(nn.Module):
    def __init__(self, model_dim: int = 256, decoder_hidden_dim: int = 64, dilation: bool = False):
        super().__init__()
        self.decoder_type = 'spatial_2d'
        scale_layer_num = 2 if dilation else 3
        self.decoder = SpatialDecoderBlock(
            model_dim, decoder_hidden_dim, scale_layer_num, 1, 2, '2D')

    def forward(self, feature: Tensor, frame_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feature: [\sigma t_i, c, h, w]
            frame_mask: [\sigma t_i, h, w]
        Return:
            heatmap: [\sigma t_i, 1, h, w]
            regression: [\sigma ti, 2, h, w]
        """
        heatmap, regression = self.decoder(feature)
        heatmap = heatmap.sigmoid()
        if frame_mask is not None:
            frame_mask = F.interpolate(
                frame_mask[:, None].float(), heatmap.shape[-2:], mode='nearest').bool()
            heatmap = heatmap.masked_fill(frame_mask, 0.)
            regression = regression.masked_fill(frame_mask, 0.)
        return {
            'spatial_map': heatmap,
            'spatial_wh': regression
        }


class TemporalDecoderAnchor(nn.Module):
    def __init__(self, model_dim: int = 256, temporal_window_width: Optional[List] = None, dropout: float = 0.1):
        super().__init__()
        self.decoder_type = 'temporal_anchor'
        self.temporal_window_width = temporal_window_width
        self.reg_head = MLP(2*model_dim, model_dim,
                            len(temporal_window_width) * 2, 2, dropout=dropout)
        self.cls_head = MLP(2*model_dim, model_dim,
                            len(temporal_window_width), 2, dropout=dropout)

    def forward(self, feature_global: Tensor, feature_obj: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feature_global: [b, c, t]
            feature_obj: [b, c, t]
        Return:
            offset: [b, t*n_window, 2]
            pred_score: [b, t*n_window]
        """

        feature_flatten = torch.cat(
            [feature_obj, feature_global], dim=1).transpose(1, 2)   # [b, t, 2*c]
        offset = self.reg_head(feature_flatten)  # [b, t, 2*n_window]
        offset = offset.contiguous().view(-1,
                                          offset.shape[1] * len(self.temporal_window_width), 2)  # [b, t*n_window, 2]
        pred_score = self.cls_head(feature_flatten)  # [b, t, n_window]
        pred_score = torch.sigmoid(pred_score).contiguous().view(
            pred_score.size(0), -1)  # [b, t*n_window]
        return {
            'temporal_offset': offset,
            'temporal_score': pred_score
        }


class TemporalDecoderRegression(nn.Module):
    def __init__(self, model_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.decoder_type = 'temporal_regression'
        self.score_head = MLP(2*model_dim, model_dim, 1, 2, dropout=dropout)
        self.iou_head = MLP(2*model_dim, model_dim, 1, 2, dropout=dropout)
        self.reg_head = MLP(2*model_dim, model_dim, 2, 2, dropout=dropout)

    def forward(self, feature_global: Tensor, feature_obj: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feature_global: [b, c, t]
            feature_obj: [b, c, t]
        Return:
            pred_score: [b, t]
            pred_reg: [b, t, 2]
            pred_iou: [b, t]
        """

        feature_flatten = torch.cat(
            [feature_obj, feature_global], dim=1).transpose(1, 2)   # [b, t, 2*c]
        pred_score = self.score_head(
            feature_flatten).squeeze(-1).sigmoid()  # [b, t]
        pred_reg = self.reg_head(feature_flatten)  # [b, t, 2]
        pred_iou = self.iou_head(
            feature_flatten).squeeze(-1).sigmoid()  # [b, t]
        return {
            'temporal_score': pred_score,
            'temporal_reg': pred_reg,
            'temporal_iou': pred_iou,
        }


class MLP(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, output_dim: int = 256,
                 num_layers: int = 2, normalization: bool = True, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if normalization:
            self.layers = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(n, k),
                    nn.LayerNorm(k),
                ) if idx < self.num_layers - 1 else nn.Linear(n, k)
                for idx, (n, k) in enumerate(zip([input_dim] + h, h + [output_dim]))
            )
        else:
            self.layers = nn.ModuleList(
                nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
            )
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if self.dropout and i < self.num_layers:
                x = self.dropout(x)
        return x
