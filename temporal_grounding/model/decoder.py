import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import _sigmoid, score2d_to_moments_scores, nms, nms_detections
from .module.tanmodule import SparseMaxPool, Predictor
from model.module.attention import TransformerDecoder
from utils.losses import TanLoss, IoULoss, TAGLoss
import matplotlib.pyplot as plt
from dataset.generate_anchor import *

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
            loss = loss_reg + loss_iou + loss_score + weights_loss[-1] * loss_weight
            # loss = loss_reg + loss_score
            # fig1 = plt.figure()
            # plt.plot(pred_score[0].detach().cpu().numpy())
            # plt.close()
            # fig2 = plt.figure()
            # plt.plot(score_nm[0].detach().cpu().numpy())
            # plt.close()

            return None, {'loss': loss, 'loss_cls': loss_score, 'loss_reg': loss_reg}

        elif mode == 'val':
            # boxes = offset + proposals  # b*n*2
            # pred_score = pred_score  # b*n
            pred_reg = pred_reg.permute(0, 2, 1)  # b*l*2
            pred_start = index - pred_reg[:, :, 0]
            pred_end = index + pred_reg[:, :, 1]
            predictions = torch.stack([pred_start, pred_end], dim=-1)  # b*l*2
            _, indices = torch.topk(pred_score, 1, 1)
            # predict_box = torch.gather(
            #     proposals, 1, indices.unsqueeze(-1).repeat(1, 1, 2))  # b*n*2
            # predict_reg = torch.gather(
            #     offset, 1, indices.unsqueeze(-1).repeat(1, 1, 2))  # b*n*2
            boxes = torch.gather(
                predictions, 1, indices.unsqueeze(-1).repeat(1, 1, 2))  # b*n*2
            return boxes, pred_score, weights
        else:
            pred_reg = pred_reg.permute(0, 2, 1)  # b*l*2
            pred_start = index - pred_reg[:, :, 0]
            pred_end = index + pred_reg[:, :, 1]
            predictions = torch.stack([pred_start, pred_end], dim=-1)  # b*l*2
            return predictions, pred_score*pred_iou.squeeze(1).sigmoid()

            # boxes = self.inference(index, pred_score, pred_reg, pred_iou)
            # return [boxes[0][i, :].unsqueeze(0).cpu().numpy() for i in range(boxes[0].shape[0])]


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
        # self.mine = MINE(len(config['window_width'])*config['segment_num'])
        self.bce_loss = nn.BCELoss()
        self.regloss = nn.SmoothL1Loss()
        self.iou_loss = IoULoss()
        self.tagloss = TAGLoss()
        self.kl_loss = nn.KLDivLoss()

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
            # indices = torch.where(adj_mat > 0)
            score_pos = torch.where(score >= score.max().item(
            )*self.config['thres_adjmat'], score, torch.zeros_like(score)).unsqueeze(-1)
            # score_pos = torch.where(adj_mat > 0, torch.ones_like(
            #     score), torch.zeros_like(score)).unsqueeze(-1)
            # refine_box = refine_box[indices]
            # gt_reg = gt_reg.unsqueeze(1).repeat(1, refine_box.shape[1], 1).float()[indices]

            loss_reg = self.regloss(refine_box*score_pos, gt_reg.unsqueeze(
                1).repeat(1, refine_box.shape[1], 1).float()*score_pos)

            loss_cls = self.bce_loss(pred_score, score)
            # loss_cls = self.mine(pred_score, score)

            weights_loss = self.config['loss_weight']
            loss_weight = 0
            if self.config['with_weight_loss']:
                for weight in weights:
                    weight = torch.sigmoid(
                        weight).contiguous().view(weight.size(0), -1)
                    loss_weight += self.bce_loss(weight, score_nm)

            loss = weights_loss[0] * loss_cls + weights_loss[1] * \
                loss_reg + weights_loss[2] * loss_weight
            # fig1 = plt.figure()
            # plt.plot(pred_score[0].detach().cpu().numpy())
            # plt.close()
            # fig2 = plt.figure()
            # plt.plot(score[0].detach().data.cpu().numpy())
            # plt.close()
            # fig3 = plt.figure()
            # plt.plot(weight[0].detach().cpu().numpy())
            # plt.close()
            return None, {'reg_loss': loss_reg, 'cls_loss': loss_cls, 'weight_loss': loss_weight, 'loss': loss}
        elif mode == 'val':
            # boxes = offset + proposals  # b*n*2
            # pred_score = pred_score  # b*n
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


class TANDecoder(nn.Module):
    def __init__(self, config):
        super(TANDecoder, self).__init__()
        self.config = config
        self.to_2dmap = SparseMaxPool(
            self.config['pooling_counts'], self.config['segment_num'])
        self.decoder = nn.Sequential(
            nn.Conv2d(config['attention_dim'],
                      config['attention_dim'], 3, 1, 1, bias=False),
            nn.GroupNorm(4, config['attention_dim']),
            nn.ReLU(),
            nn.Conv2d(config['attention_dim'], 1, 1, 1)
        )
        # self.decoder = Predictor(config['attention_dim'], config['attention_dim'], config['tan_decoder_kernel_size'], 4, mask2d)
        self.tanloss = TanLoss(
            config['tan_min_iou'], config['tan_max_iou'], self.to_2dmap.mask2d)
        self.bce_loss = nn.BCELoss()

    def forward(self, fea, weights, tan_map, duration, mode, score_nm):
        fea = self.to_2dmap(fea)
        out = self.decoder(fea)
        if mode == 'train':
            loss_final = self.tanloss(out, tan_map)
            loss_weight = 0
            if self.config['with_weight_loss']:
                for weight in weights:
                    # weight = F.interpolate(weight, fea.shape[2:], mode='bilinear', align_corners=True)
                    weight = torch.sigmoid(
                        weight).contiguous().view(weight.size(0), -1)
                    loss_weight += self.bce_loss(weight, score_nm)
            weights_loss = self.config['loss_weight']
            loss = weights_loss[0] * loss_final + weights_loss[2] * loss_weight
            out_show = (torch.sigmoid(
                out) * self.to_2dmap.mask2d).squeeze(1)[0].detach().cpu().data.numpy()
            gt_show = (
                tan_map * self.to_2dmap.mask2d).squeeze(1)[0].detach().cpu().data.numpy()
            fig1 = plt.figure()
            plt.imshow(out_show)
            plt.close()
            fig2 = plt.figure()
            plt.imshow(gt_show)
            plt.close()
            return {'pre': fig1, 'gt': fig2}, {'loss': loss, "weight_loss": loss_weight}
        else:
            map = torch.sigmoid(out) * self.to_2dmap.mask2d
            # print(map.shape)
            map = map.squeeze()
            candidates, scores = score2d_to_moments_scores(
                map, self.config['segment_num'], duration)
            moments = nms(candidates, scores, topk=5, thresh=0.5)
            return moments.cpu()


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


class MINE(nn.Module):
    def __init__(self, anchors=1, hidden_size=10):
        super(MINE, self).__init__()
        self.layers = nn.Sequential(nn.Linear(2*anchors, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x, y):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x, ], dim=0)
        idx = torch.randperm(batch_size)

        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)

        inputs = torch.cat([tiled_x, concat_y], dim=1)

        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        loss = -(torch.mean(pred_xy)
                 - torch.log(torch.mean(torch.exp(pred_x_y))))

        return loss
