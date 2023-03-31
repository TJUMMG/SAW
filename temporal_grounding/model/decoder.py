import torch
import torch.nn as nn
# from dataset.generate_anchor import *


class RegressionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.score_head = nn.Sequential(
            nn.LayerNorm(config['attention_dim']),
            nn.Linear(config['attention_dim'],
                      config['attention_dim']),
            nn.LayerNorm(config['attention_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['attention_dim'], 1)
        )
        self.iou_head = nn.Sequential(
            nn.LayerNorm(config['attention_dim']),
            nn.Linear(config['attention_dim'],
                      config['attention_dim']),
            nn.LayerNorm(config['attention_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['attention_dim'], 1)
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(config['attention_dim']),
            nn.Linear(config['attention_dim'],
                      config['attention_dim']),
            nn.LayerNorm(config['attention_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['attention_dim'], 2)
        )
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.SmoothL1Loss()

    def compute_reg_tar(self, label, score):
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

    def forward(self, fea, weights, label, score_nm, mode='train'):
        fea = fea.permute(0, 2, 1)
        pred_score = self.score_head(fea).permute(0, 2, 1)
        pred_reg = self.reg_head(fea).permute(0, 2, 1)
        pred_iou = self.iou_head(fea).permute(0, 2, 1)
        pred_score = pred_score.squeeze(1).sigmoid()

        index = torch.arange(0, score_nm.shape[1]).unsqueeze(
            0).repeat(fea.shape[0], 1).to(pred_reg.device)
        if mode == 'train':
            pred_reg = pred_reg.permute(0, 2, 1)  # b*l*2
            pred_iou = pred_iou.squeeze(1).sigmoid()

            pred_start = index - pred_reg[:, :, 0]
            pred_end = index + pred_reg[:, :, 1]
            predictions = torch.stack(
                [pred_start, pred_end], dim=-1) / self.config['segment_num']
            predictions.clamp_(min=0, max=1)

            label_reg = self.compute_reg_tar(label, score_nm).float()
            iou_target = segment_tiou(
                predictions, label[:, None, :] / self.config['segment_num'])

            iou_pos_ind = iou_target > 0.5
            pos_iou_target = iou_target[iou_pos_ind]
            pos_iou_pred = pred_iou[iou_pos_ind]
            loss_reg = self.l1_loss(
                pred_reg * score_nm.unsqueeze(-1), label_reg)
            loss_score = self.bce_loss(pred_score, score_nm)
            weights_loss = self.config['loss_weight']
            loss_weight = 0
            if self.config['with_weight_loss']:
                for weight in weights:
                    weight = torch.sigmoid(
                        weight).contiguous().view(weight.size(0), -1)
                    loss_weight += self.bce_loss(weight, score_nm)

            if iou_pos_ind.sum().item() == 0:
                loss_iou = 0
            else:
                loss_iou = self.l1_loss(pos_iou_pred, pos_iou_target)
            loss = loss_reg + loss_iou + loss_score + \
                weights_loss[-1] * loss_weight

            return {'loss': loss, 'loss_cls': loss_score, 'loss_reg': loss_reg}

        elif mode == 'val':
            pred_reg = pred_reg.permute(0, 2, 1)  # b*l*2
            pred_start = index - pred_reg[:, :, 0]
            pred_end = index + pred_reg[:, :, 1]
            predictions = torch.stack([pred_start, pred_end], dim=-1)  # b*l*2
            _, indices = torch.topk(pred_score, 1, 1)
            boxes = torch.gather(
                predictions, 1, indices.unsqueeze(-1).repeat(1, 1, 2))  # b*n*2
            return boxes, pred_score, weights
        else:
            pred_reg = pred_reg.permute(0, 2, 1)  # b*l*2
            pred_start = index - pred_reg[:, :, 0]
            pred_end = index + pred_reg[:, :, 1]
            predictions = torch.stack([pred_start, pred_end], dim=-1)  # b*l*2
            return predictions, pred_score*pred_iou.squeeze(1).sigmoid()


class AnchorBasedDecoder(nn.Module):
    def __init__(self, config):
        super(AnchorBasedDecoder, self).__init__()
        self.config = config

        self.reg_head = nn.Sequential(
            nn.LayerNorm(config['attention_dim']),
            nn.Linear(config['attention_dim'],
                      config['attention_dim']),
            nn.LayerNorm(config['attention_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['attention_dim'], len(
                config['window_width']) * 2)
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(config['attention_dim']),
            nn.Linear(config['attention_dim'],
                      config['attention_dim']),
            nn.LayerNorm(config['attention_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['attention_dim'], len(
                config['window_width']))
        )
        self.bce_loss = nn.BCELoss()
        self.regloss = nn.SmoothL1Loss()

    def forward(self, fea, weights, score, gt_reg, score_mask, score_nm, proposals, adj_mat, mode='train'):
        # print('aaaddd')
        fea = fea.permute(0, 2, 1)

        offset = self.reg_head(fea)  # b*t*2n_a
        offset = offset.contiguous(
        ).view(-1, offset.shape[1] * len(self.config['window_width']), 2)  # b*(t*n_a)*2
        pred_score = self.cls_head(fea)  # b*(t*n_a)
        pred_score = torch.sigmoid(pred_score).contiguous().view(
            pred_score.size(0), -1) * score_mask.float()
        proposals = proposals.view(proposals.shape[0], -1, 2).float()
        if mode == 'train':
            refine_box = proposals + offset
            score_pos = torch.where(score >= score.max().item(
            )*self.config['thres_adjmat'], score, torch.zeros_like(score)).unsqueeze(-1)

            loss_reg = self.regloss(refine_box*score_pos, gt_reg.unsqueeze(
                1).repeat(1, refine_box.shape[1], 1).float()*score_pos)

            loss_cls = self.bce_loss(pred_score, score)

            weights_loss = self.config['loss_weight']
            loss_weight = 0
            if self.config['with_weight_loss']:
                for weight in weights:
                    weight = torch.sigmoid(
                        weight).contiguous().view(weight.size(0), -1)
                    loss_weight += self.bce_loss(weight, score_nm)

            loss = weights_loss[0] * loss_cls + weights_loss[1] * \
                loss_reg + weights_loss[2] * loss_weight
            return {'reg_loss': loss_reg, 'cls_loss': loss_cls, 'weight_loss': loss_weight, 'loss': loss}

        elif mode == 'val':
            _, indices = torch.topk(pred_score, 1, 1)
            predict_box = torch.gather(
                proposals, 1, indices.unsqueeze(-1).repeat(1, 1, 2))  # b*n*2
            predict_reg = torch.gather(
                offset, 1, indices.unsqueeze(-1).repeat(1, 1, 2))  # b*n*2
            boxes = predict_box + predict_reg  # b*n*2
            return boxes, pred_score, weights
        else:
            boxes = offset + proposals  # b*n*2
            return boxes, pred_score


def segment_tiou(box_a, box_b):

    # gt: [batch, 1, 2], detections: [batch, 56, 2]
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
