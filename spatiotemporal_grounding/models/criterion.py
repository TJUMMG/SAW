import torch
import torch.nn.functional as F
import torch.nn as nn
from .anchor_utils import generate_proposals, generate_scores, generate_2d_gaussian
from einops import repeat, rearrange
from .utils import generate_anchor_scores, compute_temporal_reg_tar, segment_tiou
from .utils import generate_2d_gaussian as generate_2d_gaussian_new


class SetCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.temporal_reg_loss = nn.SmoothL1Loss()
        self.temporal_cls_loss = nn.BCELoss()
        self.spatial_hm_loss = nn.BCELoss()
        self.spatial_wh_loss = nn.SmoothL1Loss()

    def loss_spatial(self, outputs, targets, inter_idx):
        inter_idx = inter_idx[0]
        h, w = outputs['spatial_map'].shape[-2:]
        box_gt = [targets[i]['boxes'] for i in range(len(targets))]
        box_gt = torch.cat(box_gt, dim=0)   # k*4, cxcywh

        size_gt = [targets[i]['size']
                   for i in range(len(targets))]  # current input frame size
        size_gt = torch.stack(size_gt)  # [\sigma t_i, 2]
        padded_size_gt = torch.max(size_gt, dim=0)[0]    # [2]
        box_gt_unnormed = box_gt * \
            torch.stack([size_gt[:, 1], size_gt[:, 0],
                        size_gt[:, 1], size_gt[:, 0]], dim=-1)
        padded_box_gt = box_gt_unnormed / \
            torch.stack([padded_size_gt[1], padded_size_gt[0],
                        padded_size_gt[1], padded_size_gt[0]], dim=-1)[None]
        gaussian_gt = generate_2d_gaussian(padded_box_gt, w, h, delta=0.05)[
            :, None]  # k*1*h*w
        wh_gt = (padded_box_gt[:, 2:] * torch.as_tensor([w, h])[None,
                 :].to(box_gt.device))[..., None, None].repeat(1, 1, h, w)

        pred_hm = outputs['spatial_map']    # k*1*h*w
        pred_wh = outputs['spatial_wh']    # k*2*h*w

        loss_hm = self.spatial_hm_loss(pred_hm, gaussian_gt)
        loss_wh = self.spatial_wh_loss(pred_wh*gaussian_gt, wh_gt*gaussian_gt)
        loss_map = 0
        for map in outputs['maps']:
            map = F.interpolate(
                map, (h, w), mode='bilinear', align_corners=True)
            loss_map += self.spatial_hm_loss(map, gaussian_gt)

        return {
            'spatial_hm_loss': loss_hm,
            'spatial_wh_loss': loss_wh,
            'spatial_map_loss': loss_map
        }, gaussian_gt

    def loss_temporal(self, outputs, durations, inter_idx):
        device = outputs['spatial_map'].device
        seq_len = max(durations)
        b = len(durations)
        inter_idx = torch.as_tensor(inter_idx).float().to(device)  # [b, 2]
        index = torch.as_tensor([i for i in range(seq_len)]).to(device)[
            None].repeat(b, 1)  # [b, t]
        inter_idx_expand = inter_idx[:, None].repeat(
            1, seq_len, 1)  # [b, t, 2]
        # [b, t], 1 for moments when action happens, otherwise 0
        action_gt = ((index >= inter_idx_expand[..., 0]) & (
            index <= inter_idx_expand[..., 1])).float()

        # [b, t] "True" represent the padded moment
        time_mask = torch.ones(b, seq_len).bool().to(device)
        for i_dur, duration in enumerate(durations):
            time_mask[i_dur, :duration] = False
        if self.cfg.temporal_decoder_type == 'anchor':
            proposals = generate_proposals(seq_len, self.cfg.temporal_window_width)[
                None].repeat(b, 1, 1).to(device)    # [b, t*n_window, 2]
            score_gt, score_mask = generate_anchor_scores(
                proposals, inter_idx, seq_len, self.cfg.temporal_score_thres)
            time_mask_expanded = repeat(
                time_mask, 'b t -> b (t n)', n=len(self.cfg.temporal_window_width))
            score_mask[time_mask_expanded] = True   # [b, t*n_window]
            score_pos = (score_gt >= self.cfg.temporal_valid_thres).float()
            score_pos = score_pos.masked_fill(time_mask_expanded, 0.)
            reg_gt = inter_idx[:, None].repeat(1, proposals.shape[1], 1)
            refined_box = outputs['temporal_offset'] + \
                proposals   # [b, t*n_window, 2]
            loss_reg = self.temporal_reg_loss(
                refined_box*score_pos[..., None], reg_gt*score_pos[..., None])
            loss_cls = self.temporal_cls_loss(outputs['temporal_score'].masked_fill(time_mask_expanded[..., None], 0.),
                                              score_gt.masked_fill(time_mask_expanded[..., None], 0.))
            return {
                'temporal_cls_loss': loss_cls,
                'temporal_align_loss': loss_reg
            }

        elif self.cfg.temporal_decoder_type == 'regression':
            pred_start = index - outputs['temporal_reg'][:, :, 0]
            pred_end = index + outputs['temporal_reg'][:, :, 1]
            predictions = torch.stack([pred_start, pred_end], dim=-1) / seq_len
            predictions = torch.clamp(predictions, 0, 1)
            label_reg = compute_temporal_reg_tar(inter_idx, action_gt)
            label_iou = segment_tiou(predictions, inter_idx[:, None] / seq_len)
            iou_pos_ind = label_iou > 0.5
            pos_iou_target = label_iou[iou_pos_ind]
            pos_iou_pred = outputs['temporal_iou'][iou_pos_ind]
            loss_reg = self.temporal_reg_loss(
                outputs['temporal_reg'] * action_gt.unsqueeze(-1), label_reg)
            loss_score = self.temporal_cls_loss(
                outputs['temporal_score'], action_gt)
            if iou_pos_ind.sum().item() == 0:
                loss_iou = 0
            else:
                loss_iou = self.temporal_cls_loss(
                    pos_iou_pred, pos_iou_target.detach())
            return {
                'temporal_score_loss': loss_score,
                'temporal_reg_loss': loss_reg,
                'temporal_iou_loss': loss_iou
            }
        else:
            raise NotImplementedError

    def forward(self, outputs, durations, inter_idx, targets):
        loss_dict = self.loss_temporal(outputs, durations, inter_idx)
        loss_dict_s, gaussian_gt = self.loss_spatial(
            outputs, targets, inter_idx)
        loss_dict.update(loss_dict_s)
        return loss_dict, gaussian_gt
