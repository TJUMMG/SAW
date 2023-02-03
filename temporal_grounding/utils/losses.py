import torch
import torch.nn as nn
import torch.nn.functional as F

def _neg_loss(pred, gt):
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()
  neg_weights = torch.pow(1 - gt, 4)

  loss = 0
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = (pos_loss).sum()
  neg_loss = (neg_loss).sum()

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


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, ind, target):
        pred = output.gather(1, ind)
        loss = F.l1_loss(pred, target, size_average=False)
        return loss

def generate_weight(target):

    pos = torch.where(target>0, torch.ones_like(target), torch.zeros_like(target))
    neg = torch.where(target==0, torch.ones_like(target), torch.zeros_like(target))
    # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    # if num_pos == 0:
    #     weights = alpha * neg
    # else:
    weights = alpha * pos + beta * neg

    return weights


class TanLoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d

    def scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores2d, ious2d):
        ious2d = self.scale(ious2d).clamp(0, 1)
        return F.binary_cross_entropy_with_logits(
            scores2d.masked_select(self.mask2d),
            ious2d.masked_select(self.mask2d)
        )

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, box_a, box_b):
        inter_max_xy = torch.min(box_a[:, -1], box_b[:, -1])
        inter_min_xy = torch.max(box_a[:, 0], box_b[:, 0])
        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)

        # calculate union
        union_max_xy = torch.max(box_a[:, -1], box_b[:, -1])
        union_min_xy = torch.min(box_a[:, 0], box_b[:, 0])
        union = torch.clamp((union_max_xy - union_min_xy), min=0)

        iou = inter / (union + 1e-6)

        return 1 - iou.mean()

class TAGLoss(nn.Module):
    def __init__(self):
        super(TAGLoss, self).__init__()

    def forward(self, net_outs, gts):

        ac_loss = (-gts*torch.log(net_outs+1e-8)).sum(1) / gts.sum(1)
        ac_loss = ac_loss.mean()

        return ac_loss