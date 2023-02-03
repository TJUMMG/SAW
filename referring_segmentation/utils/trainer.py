import torch
import torch.nn as nn
from torchvision import utils
from torch.optim import Adam, SGD, AdamW
import os
from torch.utils import data
import torch.nn.functional as F
import time
import datetime
from .loss import SSIM, IOU, dice_loss, sigmoid_focal_loss
from tqdm import tqdm
import numpy as np
from utils.utils import report_result

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.lr_backbone = config['lr_backbone']
        self.lr_branch = config['lr_branch']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.model_log = os.path.join(config['log_root'], 'checkpoints')
        self.temp_root = os.path.join(config['log_root'], 'temp')
        self.loss_log = []
        if not os.path.exists(self.model_log):
            os.mkdir(self.model_log)
        if not os.path.exists(self.temp_root):
            os.mkdir(self.temp_root)


    def create_optimizer(self, model):

        param_dicts = [
            {"params": [p for n, p in model.module.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.module.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        if self.config['optimizer'] == 'Adam':
            self.optimizer = Adam(param_dicts, lr=self.lr_branch, weight_decay=0.0005)
        elif self.config['optimizer'] == 'AdamW':
            self.optimizer = AdamW(param_dicts, lr=self.lr_branch, weight_decay=0.0005)
        elif self.config['optimizer'] == 'SGD':
            self.optimizer = SGD(param_dicts, lr=self.lr_branch, weight_decay=0.0005)
        else:
            raise NotImplementedError
        
    def create_lr_schedule(self, optimizer, last_epoch):
        if self.config['lr_schedule'] == 'Step':
            self.lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2, last_epoch=last_epoch)
        elif self.config['lr_schedule'] == 'Multi_step':
            self.lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1, last_epoch=last_epoch)
        else:
            raise NotImplementedError

    def model_info(self):
        print(self.model)
        num_params = 0
        num_train_param = 0
        for p in self.model.parameters():
            num_params += p.numel()  # 返回一个tensor变量内所有元素个数
            if p.requires_grad:
                num_train_param += p.numel()
        print("The total number of parameters: {}, training parameters number: {}".format(num_params, num_train_param))

    def train_one_epoch(self, model, train_loader, epoch):
        model.train()
        for i, data_batch in enumerate(train_loader):

            data_batch = sorted_batch(data_batch)
            frames, labels, embeddings = data_batch['frames'], data_batch['label'], data_batch['word_embedding']
            is_annotated = data_batch['is_annotated']
            embedding_length = data_batch['embedding_length']
            if self.config['cuda']:
                for f in range(len(frames)):
                    frames[f] = frames[f].cuda()
                    labels[f] = labels[f].cuda()
                embeddings = embeddings.cuda()
            self.optimizer.zero_grad()
            predictions, maps, _, _ = model(frames, embeddings, embedding_length)
            save_frame = torch.cat([f[0].unsqueeze(0) for f in frames], dim=0)
            save_pre = torch.cat([f[0].unsqueeze(0) for f in predictions], dim=0)
            save_label = torch.cat([f[0].unsqueeze(0) for f in labels], dim=0)

            loss = []
            wbce_loss = 0
            n_bce_item = 0

            for j, prediction in enumerate(predictions):
                for b in range(self.batch_size):
                    if is_annotated[j][b]:
                        if len(maps) != 0:
                            label_low = F.interpolate(labels[j][b].unsqueeze(0), maps[0][j][b].shape[1:], mode='nearest')
                        if self.config['loss_function'] == 'BCE':
                            loss.append(F.binary_cross_entropy_with_logits(prediction[b].unsqueeze(0), labels[j][b].unsqueeze(0), reduction='mean'))
                            for map in maps:
                                loss.append(self.config['lr_weight']*F.binary_cross_entropy_with_logits(map[j][b].unsqueeze(0), label_low, reduction='mean'))
                        elif self.config['loss_function'] == 'weightedBCE':
                            weight = generate_weight(labels[j][b].unsqueeze(0))
                            if len(maps) != 0:
                                weight_low = generate_weight(label_low)
                            loss.append(F.binary_cross_entropy_with_logits(prediction[b].unsqueeze(0), labels[j][b].unsqueeze(0), weight=weight, reduction='mean'))
                            for map in maps:
                                loss.append(self.config['lr_weight']*F.binary_cross_entropy_with_logits(map[j][b].unsqueeze(0), label_low, weight=weight_low, reduction='mean'))
                        elif self.config['loss_function'] == 'Dice':
                            dice = BinaryDiceLoss()
                            if len(maps) != 0:
                                label_low = F.interpolate(labels[j][b].unsqueeze(0), maps[0][j][b].shape[1:],
                                                          mode='nearest')
                                weight_low = generate_weight(label_low)
                            loss.append(dice(torch.sigmoid(prediction[b]).unsqueeze(0), labels[j][b].unsqueeze(0)))
                            for map in maps:
                                loss.append(self.config['lr_weight'] * F.binary_cross_entropy_with_logits(
                                    map[j][b].unsqueeze(0), label_low, weight=weight_low, reduction='mean'))
                        elif self.config['loss_function'] == 'SSIM':
                            iou = IOU()
                            ssimloss = SSIM()
                            weight = generate_weight(labels[j][b].unsqueeze(0))
                            if len(maps) != 0:
                                weight_low = generate_weight(label_low)
                            loss.append(F.binary_cross_entropy_with_logits(
                                prediction[b].unsqueeze(0), labels[j][b].unsqueeze(0), weight=weight, reduction='mean') + iou(torch.sigmoid(prediction[b].unsqueeze(0)),
                                                                                          labels[j][b].unsqueeze(0)) + (
                                                    1 - ssimloss(torch.sigmoid(prediction[b].unsqueeze(0)), labels[j][b].unsqueeze(0))))
                            # wbce_loss += F.binary_cross_entropy_with_logits(
                            #     prediction[b].unsqueeze(0), labels[j][b].unsqueeze(0), weight=weight, reduction='mean')
                            n_bce_item += 1

                            for map in maps:
                                loss.append(self.config['lr_weight'] * F.binary_cross_entropy_with_logits(
                                    map[j][b].unsqueeze(0), label_low, weight=weight_low, reduction='mean'))
                        else:
                            raise NotImplementedError
            # with open('./logs/loss_wbce.txt', 'a+') as f:
            #     f.write(str(wbce_loss.data.cpu().numpy() / n_bce_item) + '\n')
            loss = sum(loss) / self.batch_size
            loss.backward()
            self.optimizer.step()
            if i % self.config['save_iters'] == 0:

                print('epoch: {}/{} | iter: {}/ {} | loss: {} | lr: {}'.format(epoch, self.epochs, i, len(train_loader), loss, self.lr_schedule.get_last_lr()))
                utils.save_image(save_frame, os.path.join(self.temp_root, 'iter_{}_frame.png'.format(i)), padding=0)
                utils.save_image(save_label, os.path.join(self.temp_root, 'iter_{}_label.png'.format(i)), padding=0)
                # utils.save_image(torch.sigmoid(save_map), os.path.join(self.temp_root, 'iter_{}_map.png'.format(i)), padding=0)
                utils.save_image(torch.sigmoid(save_pre), os.path.join(self.temp_root, 'iter_{}_prediction.png'.format(i)), padding=0)
        self.lr_schedule.step()
        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'lr_schedule': self.lr_schedule.state_dict()}, os.path.join(self.model_log, 'checkpoint{}.pth'.format(epoch)))

    def train_one_epoch_backbone(self, model, train_loader, epoch):
        for i, data_batch in enumerate(train_loader):
            frame, label = data_batch['frame'], data_batch['label']
            if self.config['cuda']:
                frame, label = frame.cuda(), label.cuda()
            self.optimizer.zero_grad()
            prediction = model(frame)

            loss = F.binary_cross_entropy_with_logits(prediction, label)
            loss = loss / self.batch_size
            loss.backward()
            self.optimizer.step()
            if i % self.config['save_iters'] == 0:
                print('epoch: {}/{} | iter: {}/ {} | loss: {} | lr: {}'.format(epoch, self.epochs, i, len(train_loader), loss, self.lr_schedule.get_last_lr()))
                utils.save_image(frame, os.path.join(self.temp_root, 'iter_{}_frame.png'.format(i)), padding=0)
                utils.save_image(label, os.path.join(self.temp_root, 'iter_{}_label.png'.format(i)), padding=0)
                utils.save_image(torch.sigmoid(prediction), os.path.join(self.temp_root, 'iter_{}_prediction.png'.format(i)), padding=0)

        self.lr_schedule.step()
        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'lr_schedule': self.lr_schedule.state_dict()},
                       os.path.join(self.model_log, 'checkpoint{}.pth'.format(epoch+1)))

    def val(self, model, loader):
        model.eval()

        num_frames = 0
        total_times = 0
        pres = []
        gts = []

        pres_s = []
        pres_m = []
        pres_l = []
        gts_s = []
        gts_m = []
        gts_l = []

        with torch.no_grad():
            print('video sequence num: {}'.format(len(loader)))
            print('testing.....')

            for data_batch in tqdm(loader):
                frames, labels, embedding = data_batch['frames'], data_batch['label'], data_batch['word_embedding']
                embedding_length = data_batch['embedding_length']
                is_annotated = data_batch['is_annotated']
                video = data_batch['video']
                name = data_batch['name']
                instance = data_batch['instance']
                # if not os.path.exists(os.path.join(self.save_fold, video[0])):
                #     os.mkdir(os.path.join(self.save_fold, video[0]))
                #     os.mkdir(os.path.join(self.save_fold, video[0], 'pre'))
                #     os.mkdir(os.path.join(self.save_fold, video[0], 'gt'))
                # video_save_root = os.path.join(self.save_fold, video[0], 'pre')
                # gt_save_fold = os.path.join(self.save_fold, video[0], 'gt')
                if self.config['cuda']:
                    for f in range(len(frames)):
                        frames[f] = frames[f].cuda()
                        labels[f] = labels[f].cuda()
                    embedding = embedding.cuda()
                num_frames += len(frames)
                start_time = time.time()
                predictions, maps, _, _ = model(frames, embedding, embedding_length)

                for j, prediction in enumerate(predictions):
                    if is_annotated[j][0] == 1:
                        pre = torch.sigmoid(prediction)
                        pre = (pre - pre.min()) / (pre.max() - pre.min())
                        pre = F.interpolate(pre, labels[j].shape[2:], mode='bilinear', align_corners=True)
                        pre_thres = torch.where(pre>0.5, torch.ones_like(pre), torch.zeros_like(pre))
                        gts.append(labels[j][0][0].cpu().numpy().astype(np.uint8))
                        pres.append(pre_thres[0][0].cpu().numpy().astype(np.uint8))
                        # if len(predictions) > 80 and len(predictions) < 150:
                        #     gts_m.append(labels[j][0][0].cpu().numpy().astype(np.uint8))
                        #     pres_m.append(pre_thres[0][0].cpu().numpy().astype(np.uint8))
                        # if len(predictions) > 100:
                        #     gts_l.append(labels[j][0][0].cpu().numpy().astype(np.uint8))
                        #     pres_l.append(pre_thres[0][0].cpu().numpy().astype(np.uint8))
                        # else:
                        #     gts_s.append(labels[j][0][0].cpu().numpy().astype(np.uint8))
                        #     pres_s.append(pre_thres[0][0].cpu().numpy().astype(np.uint8))

                total_times += time.time() - start_time


        total_times = datetime.timedelta(seconds=int(total_times))
        time_per_frame = total_times / num_frames

        print('prediction time per frame: {}s'.format(time_per_frame))

        print('evaluation...')
        meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP = report_result(pres, gts)
        print('evaluation results: meanIOU: {} | overallIOU: {} | P@5: {} | P@6: {} | P@7: {} | P@8: {} | P@9: {} | mAP: {}'.format(meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP))

        # meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP = report_result(pres_s, gts_s)
        # print(
        #     'evaluation short results: meanIOU: {} | overallIOU: {} | P@5: {} | P@6: {} | P@7: {} | P@8: {} | P@9: {} | mAP: {}'.format(
        #         meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP))

        # # meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP = report_result(pres_m, gts_m)
        # # print(
        # #     'evaluation middle results: meanIOU: {} | overallIOU: {} | P@5: {} | P@6: {} | P@7: {} | P@8: {} | P@9: {} | mAP: {}'.format(
        # #         meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP))

        # meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP = report_result(pres_l, gts_l)
        # print(
        #     'evaluation long results: meanIOU: {} | overallIOU: {} | P@5: {} | P@6: {} | P@7: {} | P@8: {} | P@9: {} | mAP: {}'.format(
        #         meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP))
        return meaIOU

    def train(self, model, dataset, val_dataset):
        self.model = model

        self.create_optimizer(model)
        self.create_lr_schedule(self.optimizer, -1)
        self.model_info()
        start_epoch = 0
        if self.config['resume'] != '':
            assert os.path.exists(self.config['resume']), ('checkpoint not exist')
            print('loading checkpoint ....')
            checkpoint = torch.load(self.config['resume'])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_schedule.load_state_dict(checkpoint['lr_schedule'])
            start_epoch = checkpoint['epoch'] + 1

        train_loader = data.DataLoader(dataset, self.batch_size, shuffle=True, num_workers=self.config['num_worker'], drop_last=True)
        val_loader = data.DataLoader(val_dataset, 1, False, num_workers=8)
        model.zero_grad()
        start_time = time.time()

        m_IoU_max = 0
        for epoch in range(start_epoch, self.epochs):
            if self.config['train_backbone']:
                self.train_one_epoch_backbone(model, train_loader, epoch)
            else:
                self.train_one_epoch(model, train_loader, epoch)

            if (epoch + 1) % 4 == 0:
                mIoU = self.val(model, val_loader)
                if mIoU > m_IoU_max:
                    torch.save({'epoch': epoch,
                            'state_dict': model.module.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'lr_schedule': self.lr_schedule.state_dict()}, os.path.join(self.model_log, 'checkpoint_best.pth'.format(epoch)))


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        
def sorted_batch(data_batch):
    embedding_length = data_batch['embedding_length']
    new_length, index = torch.sort(embedding_length, descending=True)
    batch = data_batch['word_embedding'].shape[0]
    new_isannotated = []
    new_frames = []
    new_labels = []
    for i in range(len(data_batch['frames'])):
        new_frames.append(torch.zeros_like(data_batch['frames'][i]))
        new_labels.append(torch.zeros_like(data_batch['label'][i]))
        new_isannotated.append(torch.zeros_like(data_batch['is_annotated'][i]))
    new_embedding = torch.zeros_like(data_batch['word_embedding'])
    for b in range(batch):
        new_embedding[b] = data_batch['word_embedding'][index[b]]
        for j in range(len(data_batch['frames'])):
            new_frames[j][b] = data_batch['frames'][j][index[b]]
            new_labels[j][b] = data_batch['label'][j][index[b]]
            new_isannotated[j][b] = data_batch['is_annotated'][j][index[b]]
    data_batch['word_embedding'] = new_embedding
    data_batch['embedding_length'] = new_length
    data_batch['frames'] = new_frames
    data_batch['label'] = new_labels
    data_batch['is_annotated'] = new_isannotated
    return data_batch


def generate_weight(target):

    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
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


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss