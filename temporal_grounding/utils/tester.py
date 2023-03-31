import torch
import os
import time
import datetime
from torch.utils import data
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.utils import CountMeter, compute_IoU_recall
import collections


class Tester(object):
    def __init__(self, config):
        self.config = config
        self.checkpoint = config['checkpoint']
        assert os.path.exists(self.checkpoint), 'incorrect checkpoint path!'

    def model_info(self):
        print(self.model)
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()  # 返回一个tensor变量内所有元素个数
        print("The total number of parameters: {}".format(num_params))

    def test(self, model, dataset):
        self.model = model
        self.model.eval()
        self.model_info()
        print('loading checkpoint ....')
        checkpoint = torch.load(self.checkpoint)
        self.model.module.load_state_dict(checkpoint['state_dict'])
        loader = data.DataLoader(
            dataset, self.config['batch_size'], False, num_workers=8)
        meters_5 = collections.defaultdict(lambda: CountMeter())
        recall_metrics = (1, 5)
        iou_metrics = (0.1, 0.3, 0.5, 0.7)
        table = [['Rank@{},mIoU@{}'.format(i, j)
                  for i in recall_metrics for j in iou_metrics]]

        for i, data_batch in tqdm(enumerate(loader), total=len(loader)):
            fea, embedding, score, score_mask, embedding_length, label, proposals, score_nm, adj_mat = \
                data_batch['feat'], data_batch['embedding'], data_batch['score'], data_batch['score_mask'], \
                data_batch['embedding_length'], data_batch['label'], data_batch['proposals'], data_batch['score_nm'], \
                data_batch['adj_mat']
            if self.config['cuda']:
                fea, embedding, score, score_mask, label, proposals, score_nm, adj_mat = \
                    fea.cuda(), embedding.cuda(), score.cuda(), score_mask.cuda(), label.cuda(), proposals.cuda(), \
                    score_nm.cuda(), adj_mat.cuda()

            with torch.no_grad():
                predict_boxes, score = self.model(fea, embedding, embedding_length, score,
                                                  label, score_mask, score_nm, proposals, adj_mat, 'test')
                predict_boxes_old = np.round(
                    predict_boxes.cpu().numpy()).astype(np.int32)
                for k in range(predict_boxes.shape[0]):
                    gt_boxes = label[k]
                    predict_boxes = predict_boxes_old[k]
                    predict_flatten = score[k]
                    gt_starts, gt_ends = gt_boxes[0], gt_boxes[1]
                    predict_starts, predict_ends = predict_boxes[:,
                                                                 0], predict_boxes[:, 1]
                    predict_starts[predict_starts < 0] = 0
                    seq_len = self.config['segment_num']
                    predict_ends[predict_ends >= seq_len] = seq_len - 1
                    predict_flatten = predict_flatten.cpu().numpy()
                    predict_boxes[:, 0], predict_boxes[:,
                                                       1] = predict_starts, predict_ends

                    topn_IoU_matric = compute_IoU_recall(
                        predict_flatten, predict_boxes, gt_boxes)
                    meters_5['mIoU'].update(topn_IoU_matric, 1)

        IoU_threshs = [0.1, 0.3, 0.5, 0.7]
        top_n_list = [1, 5]
        topn_IoU_matric, count = meters_5['mIoU'].val, meters_5['mIoU'].count
        for i in range(2):
            for j in range(4):
                print('{}, {:.4f}'.format('IoU@' + str(top_n_list[i]) + '@' + str(IoU_threshs[j]),
                                          topn_IoU_matric[i, j] / count), end=' | ')
