import json
import os
from tqdm import tqdm
import cv2
import h5py
import numpy as np


def generate_split(dataset, root='/media/HardDisk/wwk/video_text/datasets/', saveroot='/media/HardDisk/wwk/video_text/datasets/Charades-STA/split'):
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    if dataset == 'Charades-STA':
        with open(os.path.join(root, 'charades_movie_length_info.txt'),
                  'r') as f:
            lines = f.readlines()
            duration_info = dict()
            for line in tqdm(lines, desc='read video duration'):
                line = line.split()
                duration_info[line[0]] = float(line[1])
        # duration_info = json.load(open(os.path.join(root, 'Charades_duration.json')))
        modes = ['train', 'test']
        for mode in modes:
            datas = {}
            file = os.path.join(root, 'charades_sta_{}.txt'.format(mode))
            with open(file, 'r') as f:
                data = f.readlines()
            for line in tqdm(data):
                line = line.strip()
                name = line.split(' ')[0]
                moment = [float(line.split(' ')[1]), float(
                    line.split(' ')[2].split('##')[0])]
                if name not in datas.keys():
                    datas[name] = {}
                    datas[name]['duration'] = duration_info[name]
                    n_num = 0
                    datas[name]['label'] = {}
                    datas[name]['label'][str(n_num)] = moment
                else:
                    n_num += 1
                    datas[name]['label'][str(n_num)] = moment

            with open(os.path.join(saveroot, '{}.json'.format(mode)), 'w+') as f:
                json.dump(datas, f, indent=1)
    elif dataset == 'ActivityNet':
        train_json = {}
        val_json = {}
        test_json = {}
        train_infos = json.load(open(os.path.join(root, 'train.json')))
        val_1_infos = json.load(open(os.path.join(root, 'val_1.json')))
        val_2_infos = json.load(open(os.path.join(root, 'val_2.json')))
        for key, value in train_infos.items():
            train_json[key] = {}
            train_json[key]['duration'] = value['duration']
            train_json[key]['label'] = {}
            for i in range(len(value['timestamps'])):
                train_json[key]['label'][str(i)] = value['timestamps'][i]

        for key, value in val_1_infos.items():
            val_json[key] = {}
            val_json[key]['duration'] = value['duration']
            val_json[key]['label'] = {}
            for i in range(len(value['timestamps'])):
                val_json[key]['label'][str(i)] = value['timestamps'][i]

        for key, value in val_2_infos.items():
            test_json[key] = {}
            test_json[key]['duration'] = value['duration']
            test_json[key]['label'] = {}
            if key in val_json.keys():
                if len(val_json[key]['label'].keys()) > 0:
                    n = len(val_json[key]['label'].keys())
                else:
                    n = 0
            for i in range(len(value['timestamps'])):
                test_json[key]['label'][str(n + i)] = value['timestamps'][i]

        json.dump(train_json, open(os.path.join(
            saveroot, 'train.json'), 'w+'), indent=1)
        json.dump(val_json, open(os.path.join(
            saveroot, 'val.json'), 'w+'), indent=1)
        json.dump(test_json, open(os.path.join(
            saveroot, 'test.json'), 'w+'), indent=1)
    elif dataset == 'TACoS':
        train_json = {}
        test_json = {}
        val_json = {}
        train_infos = json.load(open(os.path.join(root, 'train.json')))
        val_infos = json.load(open(os.path.join(root, 'val.json')))
        test_infos = json.load(open(os.path.join(root, 'test.json')))
        for key, value in train_infos.items():
            train_json[key] = {}
            train_json[key]['duration'] = value['num_frames'] / value['fps']
            train_json[key]['label'] = {}
            for i in range(len(value['timestamps'])):
                train_json[key]['label'][str(i)] = list(
                    np.array(value['timestamps'][i]) / value['fps'])

        for key, value in val_infos.items():
            val_json[key] = {}
            val_json[key]['duration'] = value['num_frames'] / value['fps']
            val_json[key]['label'] = {}
            for i in range(len(value['timestamps'])):
                val_json[key]['label'][str(i)] = list(
                    np.array(value['timestamps'][i]) / value['fps'])

        for key, value in test_infos.items():
            test_json[key] = {}
            test_json[key]['duration'] = value['num_frames'] / value['fps']
            test_json[key]['label'] = {}
            for i in range(len(value['timestamps'])):
                test_json[key]['label'][str(i)] = list(
                    np.array(value['timestamps'][i]) / value['fps'])

        json.dump(train_json, open(os.path.join(
            saveroot, 'train.json'), 'w+'), indent=1)
        json.dump(val_json, open(os.path.join(
            saveroot, 'val.json'), 'w+'), indent=1)
        json.dump(test_json, open(os.path.join(
            saveroot, 'test.json'), 'w+'), indent=1)


if __name__ == '__main__':
    # generate_split(dataset='TACoS', root='/media/HardDisk/wwk/video_text/datasets/TACoS/split/captions/', saveroot='/media/HardDisk/wwk/video_text/datasets/TACoS/split')
    generate_split('ActivityNet', '/media/HardDisk/wwk/video_text/datasets/ActivityNet/split/captions',
                   '/media/HardDisk/wwk/video_text/datasets/ActivityNet/split')
