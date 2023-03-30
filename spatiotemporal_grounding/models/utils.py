from torch import Tensor, tensor
from typing import List
import torch
import torch.nn.functional as F


def temporal_stacked_to_separate(feature: Tensor, durations: List) -> Tensor:
    """
    Args: 
        feature: [\sigma t_i *]
        durations: [t_1, t_2, ..., t_b]
    Return:
        out_feature: [b, t, *]
    """
    shape_len = len(feature.shape) - 1
    max_seq_len = max(durations)
    padding_value = []
    for i in range(shape_len):
        padding_value += [0, 0]
    feature_splits = feature.split(durations, dim=0)
    out_feature = torch.stack([F.pad(f, padding_value + [0, max_seq_len-len(f)])
                              for f in feature_splits])    # [b, t, *]
    return out_feature


def temporal_separate_to_stack(feature: Tensor, durations: List) -> Tensor:
    """
    Args: 
        feature: [b, t, *]
        durations: [t_1, t_2, ..., t_b]
    Return:
        out_feature: [\sigma t_i *]
    """
    out_feature = torch.cat([feature[i][:durations[i]]
                            for i in range(len(feature))], dim=0)  # [\sigma t_i *]
    return out_feature


def generate_anchor_scores(proposals, label, seq_len, thres_score):
    """
    Args: 
        proposals: [b, t*n_windows, 2]
        label: [b, 2]
    Return:
        scores: [b, t*n_windows]
        scores_mask: [b, t*n_windows]
    """
    illegal = torch.logical_or(
        proposals[..., 0] < 0, proposals[..., 1] >= seq_len)
    label = label[:, None].repeat(1, proposals.shape[1], 1)
    IoUs = calculate_IoU_batch_temporal(proposals, label)
    IoUs[illegal] = 0.0
    max_IoU = torch.max(IoUs, dim=1)[0]
    IoUs[IoUs < thres_score * max_IoU[:, None]] = 0.0
    IoUs = IoUs / (max_IoU[:, None] + 1e-4)
    scores = IoUs.float()
    scores_mask = (1 - illegal.float())
    return scores, scores_mask


def calculate_IoU_batch_temporal(box0: Tensor, box1: Tensor) -> Tensor:
    """
    Args: 
        box0: [b, n_boxes, 2]
        box1: [b, n_boxes, 2]
    Return:
        iou: [b, n_boxes]
    """
    union = (torch.min(torch.stack([box0[..., 0], box1[..., 0]], 0), 0)[
             0], torch.max(torch.stack([box0[..., 1], box1[..., 1]], 0), 0)[0])
    inter = (torch.max(torch.stack([box0[..., 0], box1[..., 0]], 0), 0)[
             0], torch.min(torch.stack([box0[..., 1], box1[..., 1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def generate_2d_gaussian(boxes, w, h, delta=0.05):
    """
    generate gaussian according to the input boxes, normalized to [0, 1] [checked]
    Args:
        boxes: [k, 4]   in the form of cxcywh
        w: the width of gaussian map
        h: the height of gaussian map
        delta: gaussian parameter
    Return:
        gaussian: [k, h, w]
    """
    n_boxes = len(boxes)
    ww = torch.linspace(0, w-1, w)
    hh = torch.linspace(0, h-1, h)
    gridh, gridw = torch.meshgrid(hh, ww)
    grid = torch.stack([gridw, gridh], dim=0)[None, ...].repeat(
        n_boxes, 1, 1, 1).to(boxes.device)  # [k, 2, h, w]
    boxes = boxes[..., None, None].repeat(1, 1, h, w)
    gaussian = torch.exp(-(boxes[:, 0]-grid[:, 0])**2/(delta*boxes[:, 2]**2)) *\
        torch.exp(-(boxes[:, 1]-grid[:, 1])**2 /
                  (delta*boxes[:, 3]**2))  # [k, h, w]
    gaussian[gaussian < 0.05] = 0
    return gaussian


def compute_temporal_reg_tar(label, score):
    """
    Args:
        label: [b, 2]
        score: [b, t]
    Return:
        label_reg: [b, t, 2]
    """
    label = label.unsqueeze(1)
    segment_num = score.shape[1]
    index_s = torch.arange(0, segment_num).unsqueeze(
        0).unsqueeze(-1).to(score.device)
    index_e = torch.arange(0, segment_num).unsqueeze(
        0).unsqueeze(-1).to(score.device)

    label_reg_s = index_s - label[:, :, 0].unsqueeze(-1)
    label_reg_e = label[:, :, 1].unsqueeze(-1) - index_e

    label_reg = torch.cat([label_reg_s, label_reg_e], dim=-1)
    label_reg = label_reg * score.unsqueeze(-1)
    return label_reg


def segment_tiou(box_a, box_b):

    # gt: [batch, 1, 2], detections: [batch, k, 2]
    # calculate interaction
    inter_max_xy = torch.min(box_a[:, :, -1], box_b[:, :, -1])
    inter_min_xy = torch.max(box_a[:, :, 0], box_b[:, :, 0])
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)

    # calculate union
    union_max_xy = torch.max(box_a[:, :, -1], box_b[:, :, -1])
    union_min_xy = torch.min(box_a[:, :, 0], box_b[:, :, 0])
    union = torch.clamp((union_max_xy - union_min_xy), min=0)

    iou = inter / (union+1e-6)

    return iou
