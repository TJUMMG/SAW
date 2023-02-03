import torch
import torch.nn.functional as F
import torch.nn as nn
from .anchor_utils import generate_proposals, generate_scores, generate_2d_gaussian

class SetCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.temporal_reg_loss = nn.SmoothL1Loss()
        self.temporal_cls_loss = nn.BCELoss()
        # self.spatial_hm_loss = FocalLoss()
        self.spatial_hm_loss = nn.BCELoss()
        # self.spatial_hm_loss = nn.KLDivLoss()
        self.spatial_wh_loss = nn.SmoothL1Loss()

    def loss_spatial(self, outputs, targets, inter_idx):
        inter_idx = inter_idx[0]
        h, w = outputs['spatial_map'].shape[-2:]
        box_gt = [targets[i]['boxes'] for i in range(len(targets))]

        box_gt = torch.cat(box_gt, dim=0)   # k*4, cxcywh
        
        pred_hm = outputs['spatial_map']    # k*1*h*w
        pred_wh = outputs['spatial_wh']    # k*2*h*w
        gaussian_gt = generate_2d_gaussian(box_gt, w, h)[:, None]  # k*1*h*w
        wh_gt = (box_gt[:, 2:] * torch.as_tensor([w, h])[None, :].to(box_gt.device))[..., None, None].repeat(1, 1, h, w)

        loss_hm = self.spatial_hm_loss(pred_hm.sigmoid(), gaussian_gt)
        loss_wh = self.spatial_wh_loss(pred_wh*gaussian_gt, wh_gt*gaussian_gt)
        loss_map = 0
        for map in outputs['maps']:
            map = F.interpolate(map, (h, w), mode='bilinear', align_corners=True)
            loss_map += self.spatial_hm_loss(map.sigmoid(), gaussian_gt)

        return {
            'spatial_hm_loss': loss_hm,
            'spatial_wh_loss': loss_wh,
            'spatial_map_loss': loss_map
        }, gaussian_gt

    def loss_temporal(self, outputs, durations, inter_idx):
        seq_len = durations[0]
        temporal_gt = torch.as_tensor(inter_idx[0]).to(outputs['temporal_score'].device)    # 2
        proposals = generate_proposals(seq_len, self.cfg.temporal_window_width).to(outputs['temporal_score'].device)
        score_gt, score_mask = generate_scores(proposals, temporal_gt, seq_len, self.cfg.temporal_score_thres)

        refined_box = outputs['temporal_offset'][0] + proposals # (t*n_box)*2
        score_pos = torch.where(score_gt >= score_gt.max().item(
            )*self.cfg.temporal_valid_thres, score_gt, torch.zeros_like(score_gt))
        loss_reg = self.temporal_reg_loss(refined_box*score_pos[..., None], 
                        temporal_gt[None, :].repeat(refined_box.shape[0], 1).float()*score_pos[..., None])
        loss_cls = self.temporal_cls_loss(outputs['temporal_score'][0]*score_mask, score_gt)
        return {
            'temporal_cls_loss': loss_cls,
            'temporal_reg_loss': loss_reg
        }

    def forward(self, outputs, durations, inter_idx, targets):
        loss_dict = self.loss_temporal(outputs, durations, inter_idx)
        loss_dict_s, gaussian_gt = self.loss_spatial(outputs, targets, inter_idx)
        loss_dict.update(loss_dict_s)
        return loss_dict, gaussian_gt

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

        
