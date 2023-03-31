import torch
from torch.optim import Adam, SGD, AdamW
import os
import argparse
from torch.utils import data
import time
import datetime
from tqdm import tqdm
from utils.utils import AverageMeter, CountMeter, compute_IoU_recall
import collections
import numpy as np
from warmup_scheduler import GradualWarmupScheduler
from utils.scheduler import InverseSquareRootSchedule
from utils.adam_optimizer import AdamOptimizer


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.num_updates = 0
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.model_log = os.path.join(config['log_root'], 'checkpoints')
        self.loss_log = []
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--lr', default=self.lr, type=float,
                                 help='learning rate')
        self.parser.add_argument('--weight-decay', '--wd', default=1e-7, type=float, metavar='WD',
                                 help='weight decay')

        self.n_iter = 0
        if not os.path.exists(self.model_log):
            os.makedirs(self.model_log)

    def create_optimizer(self, model):
        if self.config['optimizer'] == 'Adam':
            # self.optimizer = Adam(model.parameters(),
            #                       lr=self.lr)
            AdamOptimizer.add_args(self.parser)
            args = self.parser.parse_args()
            self.optimizer = AdamOptimizer(args, list(self.model.parameters()))
        elif self.config['optimizer'] == 'AdamW':
            self.optimizer = AdamW(
                model.parameters(), lr=self.lr, weight_decay=0.00005)

        elif self.config['optimizer'] == 'SGD':
            self.optimizer = SGD(model.parameters(),
                                 lr=self.lr, weight_decay=0.0005)
        else:
            raise NotImplementedError

    def create_lr_schedule(self, optimizer, last_epoch):
        if self.config['lr_schedule'] == 'StepLR':
            step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config['decay_epochs'], gamma=0.2,
                                                      last_epoch=last_epoch)
            self.lr_schedule = GradualWarmupScheduler(
                self.optimizer, multiplier=1, total_epoch=5, after_scheduler=step_lr)
            # self.lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config['decay_epochs'], gamma=0.2,
            #                                                    last_epoch=last_epoch)
        elif self.config['lr_schedule'] == 'MultiStepLR':
            self.lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1,
                                                                    last_epoch=last_epoch)
        elif self.config['lr_schedule'] == 'warmup':
            InverseSquareRootSchedule.add_args(self.parser)
            args = self.parser.parse_args()

            self.lr_schedule = InverseSquareRootSchedule(args, self.optimizer)
        else:
            raise NotImplementedError

    def model_info(self):
        num_params = 0
        num_train_param = 0
        for p in self.model.parameters():
            num_params += p.numel()  # 返回一个tensor变量内所有元素个数
            if p.requires_grad:
                num_train_param += p.numel()
        print("The total number of parameters: {}, training parameters number: {}".format(
            num_params, num_train_param))

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        self.lr_schedule.step()
        for i, data_batch in enumerate(train_loader):
            fea, embedding, score, score_mask, embedding_length, label, proposals, score_nm, adj_mat = \
                data_batch['feat'], data_batch['embedding'], data_batch['score'], data_batch['score_mask'], \
                data_batch['embedding_length'], data_batch['label'], data_batch['proposals'], data_batch['score_nm'], \
                data_batch['adj_mat']
            if self.config['cuda']:
                fea, embedding, score, score_mask, label, proposals, score_nm, adj_mat = \
                    fea.cuda(), embedding.cuda(), score.cuda(), score_mask.cuda(), label.cuda(), proposals.cuda(), \
                    score_nm.cuda(), adj_mat.cuda()
            self.optimizer.zero_grad()
            loss = self.model(fea, embedding, embedding_length, score,
                              label, score_mask, score_nm, proposals, adj_mat, 'train')

            try:
                self.optimizer.backward(loss['loss'])
            except:
                loss['loss'].backward()
                self.optimizer.step()
            self.num_updates += 1
            try:
                curr_lr = self.lr_schedule.step_update(self.num_updates)
            except:
                curr_lr = self.lr_schedule.get_lr()
            self.n_iter += 1

            if i % self.config['save_temp_iters'] == 0:
                print('epoch: {}/{} | iter: {}/{} | loss: {:.3f} | lr: {}'.format(epoch, self.epochs, i, len(train_loader), loss['loss'],
                                                                              curr_lr[0]))
        # self.lr_schedule.step()

    def validation(self, loader):
        self.model.eval()
        meters_5 = collections.defaultdict(lambda: CountMeter())
        meters = collections.defaultdict(lambda: AverageMeter())
        print('validation...')
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
                    predict_starts, predict_ends = predict_boxes[:,
                                                                 0], predict_boxes[:, 1]
                    predict_starts[predict_starts < 0] = 0
                    seq_len = self.config['segment_num']
                    predict_ends[predict_ends >= seq_len] = seq_len - 1
                    predict_flatten = predict_flatten.cpu().numpy()
                    predict_boxes[:, 0], predict_boxes[:,
                                                       1] = predict_starts, predict_ends

                    topn_IoU_matrix = compute_IoU_recall(
                        predict_flatten, predict_boxes, gt_boxes)
                    meters_5['mIoU'].update(topn_IoU_matrix, 1)

        IoU_threshs = [0.1, 0.3, 0.5, 0.7]
        top_n_list = [1, 5]
        topn_IoU_matrix, count = meters_5['mIoU'].val, meters_5['mIoU'].count
        for i in range(2):
            for j in range(4):
                print('{}, {:.4f}'.format('R' + str(top_n_list[i]) + ' IoU=' + str(IoU_threshs[j]),
                                          topn_IoU_matrix[i, j] / count), end=' | ')
        return topn_IoU_matrix[0, 2] / count

    def train(self, model, dataset, val_dataset=None):
        self.model = model

        self.create_optimizer(self.model)
        self.create_lr_schedule(self.optimizer, -1)
        self.model_info()
        start_epoch = 0
        if self.config['resume'] != '':
            assert os.path.exists(
                self.config['resume']), ('checkpoint not exist')
            print('loading checkpoint ....')
            checkpoint = torch.load(self.config['resume'])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_schedule.load_state_dict(checkpoint['lr_schedule'])
            start_epoch = checkpoint['epoch'] + 1

        train_loader = data.DataLoader(dataset, self.batch_size, shuffle=True, num_workers=self.config['num_worker'],
                                       drop_last=True)
        if val_dataset != None:
            val_loader = data.DataLoader(
                val_dataset, self.batch_size, shuffle=False, num_workers=self.config['num_worker'])
        self.model.zero_grad()
        start_time = time.time()

        highest_iou = 0
        for epoch in range(start_epoch, self.epochs):
            self.train_one_epoch(train_loader, epoch)
            if val_dataset != None:
                iou = self.validation(val_loader)
                if iou > highest_iou:
                    highest_iou = iou
                    torch.save({'epoch': epoch,
                                'state_dict': self.model.module.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'lr_schedule': self.lr_schedule.state_dict()},
                               os.path.join(self.model_log, 'best_model.pth'.format(epoch)))
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
