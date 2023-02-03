# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Postprocessors class to transform TubeDETR output according to the downstream task"""
import imp
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .anchor_utils import generate_proposals


class PostProcessSTVG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    @torch.no_grad()
    def forward(self, outputs, frames_id=None, video_ids=None, time_mask=None):
        """
        :param outputs: must contain a key pred_sted mapped to a [B, T, 2] tensor of logits for the start and end predictions
        :param frames_id: list of B lists which contains the increasing list of frame ids corresponding to the indexes of the decoder outputs
        :param video_ids: list of B video_ids, used to ensemble predictions when video_max_len_train < video_max_len
        :param time_mask: [B, T] tensor with False on the padded positions, used to take out padded frames from the possible predictions
        :return: list of B [start_frame, end_frame] for each video
        """

        temporal_score, temporal_offset = outputs['temporal_score'], outputs['temporal_offset']
        max_length = temporal_score.shape[1]//len(self.args.temporal_window_width)
        proposals = generate_proposals(max_length, self.args.temporal_window_width).to(temporal_score.device)
        refined_boxes = proposals[None, :] + temporal_offset
        _, ind = torch.topk(temporal_score, 1, -1)
        pred_steds = torch.gather(refined_boxes, 1, ind[..., None].repeat(1, 1, 2)).squeeze(1).long()    # b*2
        
        pred_steds = pred_steds.clamp(0, max_length-1)

        frames_id = (
            torch.tensor([row + [0] * (max_length - len(row)) for row in frames_id])
            .long()
            .to(pred_steds.device)
        )  # padded up to BxT
        # get corresponding frames id from the indexes
        pred_steds = torch.gather(frames_id, 1, pred_steds)
        pred_steds = pred_steds.float()
        pred_steds[:, 1] += 1  # the end frame is excluded in evaluation

        pred_steds = pred_steds.cpu().tolist()

        return pred_steds


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        hm_s, wh_s = outputs['spatial_map'].squeeze(1), outputs['spatial_wh']

        # Find the top response in heat map
        time, height, width = hm_s.size()
        topk_scores, topk_inds = torch.topk(hm_s.view(time, -1), 1)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float() # t*1
        topk_xs = (topk_inds % width).int().float() # t*1

        pre_wh = torch.gather(wh_s.view(wh_s.shape[0], wh_s.shape[1], -1), -1,
                                   topk_inds.unsqueeze(1).repeat(1, 2, 1))   # t*2*1
        out_bbox = torch.cat([topk_xs - pre_wh[:, 0, :] / 2,
                              topk_ys - pre_wh[:, 1, :] / 2,
                              topk_xs + pre_wh[:, 0, :] / 2,
                              topk_ys + pre_wh[:, 1, :] / 2], dim=-1)  # t*4
        out_bbox[:, 2].clamp(0, width)
        out_bbox[:, 3].clamp(0, height)


        img_h, img_w = target_sizes.unbind(1)
        scale_fct_out = torch.tensor([width, height, width, height]).float().unsqueeze(0).to(out_bbox.device)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = (out_bbox / scale_fct_out) * scale_fct

        results = [{"boxes": b} for b in boxes]

        return results


def build_postprocessors(args, dataset_name=None) -> Dict[str, nn.Module]:
    postprocessors: Dict[str, nn.Module] = {"bbox": PostProcess()}

    if dataset_name:
        if dataset_name in ["vidstg", "hcstvg"]:
            postprocessors[dataset_name] = PostProcessSTVG(args)

    return postprocessors
