import torch
import os
import time
import datetime
from torch.utils import data
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import report_result
import numpy as np
import cv2


class Tester(object):
    def __init__(self, config):
        self.config = config
        self.save_fold = config['test_savefold']
        self.checkpoint = config['checkpoint']
        if not os.path.exists(self.save_fold):
            os.mkdir(self.save_fold)
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
        loader = data.DataLoader(dataset, 1, False, num_workers=8)

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
                video_save_root = os.path.join(self.save_fold, video[0], 'pre')
                gt_save_fold = os.path.join(self.save_fold, video[0], 'gt')
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
                        if len(predictions) > 100:
                            gts_l.append(labels[j][0][0].cpu().numpy().astype(np.uint8))
                            pres_l.append(pre_thres[0][0].cpu().numpy().astype(np.uint8))
                        else:
                            gts_s.append(labels[j][0][0].cpu().numpy().astype(np.uint8))
                            pres_s.append(pre_thres[0][0].cpu().numpy().astype(np.uint8))

                total_times += time.time() - start_time


        total_times = datetime.timedelta(seconds=int(total_times))
        time_per_frame = total_times / num_frames

        print('prediction time per frame: {}s'.format(time_per_frame))

        print('evaluation...')
        meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP = report_result(pres, gts)
        print('evaluation results: meanIOU: {} | overallIOU: {} | P@5: {} | P@6: {} | P@7: {} | P@8: {} | P@9: {} | mAP: {}'.format(meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP))

        meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP = report_result(pres_s, gts_s)
        print(
            'evaluation short results: meanIOU: {} | overallIOU: {} | P@5: {} | P@6: {} | P@7: {} | P@8: {} | P@9: {} | mAP: {}'.format(
                meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP))

        # meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP = report_result(pres_m, gts_m)
        # print(
        #     'evaluation middle results: meanIOU: {} | overallIOU: {} | P@5: {} | P@6: {} | P@7: {} | P@8: {} | P@9: {} | mAP: {}'.format(
        #         meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP))

        meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP = report_result(pres_l, gts_l)
        print(
            'evaluation long results: meanIOU: {} | overallIOU: {} | P@5: {} | P@6: {} | P@7: {} | P@8: {} | P@9: {} | mAP: {}'.format(
                meaIOU, overallIOU, P5, P6, P7, P8, P9, mAP))




