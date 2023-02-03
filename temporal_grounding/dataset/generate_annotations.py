import pandas as pd
import numpy as np
import json
import os
import torch
import cv2
from tqdm import tqdm

def compute_iou(label_true, n_frame, clip_size=128, step=int(0.15*128)):
    x = torch.arange(0, n_frame)
    y = x.unfold(0, clip_size, step)
    seq = torch.arange(0, y.shape[0]).unsqueeze(1).repeat([1, clip_size])
    index = torch.stack([seq, y], dim=0).view(2, -1)
    index = (index[0].long(), index[1].long())
    videos = torch.zeros((seq.shape[0], len(x)))
    videos.index_put_(index, torch.ones(y.shape[0] * y.shape[1]))
    label_select = torch.where((x >= label_true[0] - 1) & (x < label_true[1]), torch.ones_like(x), torch.zeros_like(x))
    label_select = label_select.unsqueeze(0).repeat([seq.shape[0], 1])
    iou = (label_select * videos).sum(dim=1) / videos.sum(dim=1)
    return iou


def generate_annotation(dataset='TACoS', root='/media/HardDisk/wwk/video_text/datasets/TACoS/text/index.tsv', save_root='/media/HardDisk/wwk/video_text/datasets/TACoS'):
    datas = {}
    if dataset == 'TACoS':
        data = pd.read_csv(root, sep='\t', header=None)
        for i in tqdm(range(len(data))):
            video_name = data[2][i]
            label_true = [int(data[3][i]), int(data[4][i])]
            cap = cv2.VideoCapture(os.path.join('/media/HardDisk/wwk/video_text/datasets/TACoS/video', '{}.avi'.format(video_name)))
            n_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # iou = compute_iou(label_true, n_frame)
            # label = torch.where(iou > 0.8, torch.ones_like(iou), torch.zeros_like(iou))
            # label = label.tolist()
            if video_name not in datas.keys():
                datas[video_name] = {}
            querie_index = data[0][i]
            datas[video_name][querie_index] = {}
            # datas[video_name][querie_index]['label'] = label
            datas[video_name][querie_index]['label'] = label_true
            datas[video_name][querie_index]['length'] = n_frame
    elif dataset == 'Charades-STA':
        splits = os.listdir(root)
        for split in splits:
            data = open(os.path.join(root, split), 'r')
            lines = data.readlines()
            for line in tqdm(lines):
                line = line.strip()
                name = line.split(' ')[0]
                capture = cv2.VideoCapture(
                    os.path.join('/media/HardDisk/wwk/video_text/datasets/{}/video/'.format(dataset), name + '.mp4'))
                fps = capture.get(cv2.CAP_PROP_FPS)
                n_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                capture.release()
                start_monment = float(line.split(' ')[1]) * fps
                end_monment = float(line.split(' ')[2].split('##')[0]) * fps
                monment = [int(start_monment), int(end_monment)]
                if start_monment >= end_monment:
                    print(name)
                if name not in datas.keys():
                    datas[name] = {}
                    n_num = 0
                    datas[name][str(n_num)] = {}
                    datas[name][str(n_num)]['label'] = monment
                    datas[name][str(n_num)]['length'] = n_frame
                    datas[name][str(n_num)]['fps'] = fps
                else:
                    n_num += 1
                    datas[name][str(n_num)] = {}
                    datas[name][str(n_num)]['label'] = monment
                    datas[name][str(n_num)]['length'] = n_frame
                    datas[name][str(n_num)]['fps'] = fps
    elif dataset == 'ActivityNet':
        train_files = None


    with open(os.path.join(save_root, 'label.json'), 'w+') as f:
        json.dump(datas, f, indent=1)

if __name__ == '__main__':
    generate_annotation(dataset='Charades-STA', root='/media/HardDisk/wwk/video_text/datasets/Charades-STA/text/', save_root='/media/HardDisk/wwk/video_text/datasets/Charades-STA')