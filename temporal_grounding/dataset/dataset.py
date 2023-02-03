import os
from torch.utils import data
import h5py
import json
from utils.utils import *
from tqdm import tqdm
import torch.nn as nn
from .generate_anchor import generate_proposals, generate_scores
import torch
import numpy as np
import torchtext


class MyDataset(data.Dataset):
    # vocab = torchtext.vocab.pretrained_aliases["glove.840B.300d"]()
    def __init__(self, config, mode='train'):
        super(MyDataset, self).__init__()
        self.config = config
        self.dataset = config['{}ing_datasets'.format(mode)]
        self.embedding_type = config['embedding_type']
        self.segment_num = config['segment_num']
        self.mode = mode
        self.max_embedding_length = config['embedding_length']
        print('Preparing dataset: {}'.format(self.dataset))
        self.datas = []

        with open(os.path.join('./data', self.dataset, '{}.json'.format(mode)), 'r') as f:
            videosets = json.load(f)
        for n, video in tqdm(enumerate(videosets), total=len(videosets)):
            data = {}
            data['vid'] = video[0]
            data['timestamp'] = video[2]
            data['duration'] = video[1]
            data['words'] = video[3]
            data['index'] = n
            if (data['timestamp'][1] - data['timestamp'][0]) > 0 and data['timestamp'][1] <= data['duration'] and data['timestamp'][0] <= data['duration']:
                self.datas.append(data)
        # self.feat_path = os.path.join(
        #     config['datasets_root'], self.dataset, 'video_fea', '{}_{}.hdf5'.format(self.dataset, config['video_fea_type']))
        # self.feat_path = os.path.join(
        #     config['datasets_root'], self.dataset, 'video_fea', config['video_fea_type'])
        self.feat = h5py.File(config['datasets_root'])

        self.proposals = generate_proposals(
            config['segment_num'], config['window_width'])
        embedding_name, embedding_dim = self.config['embedding_type'].split('_')[1], int(self.config['embedding_type'].split('_')[2])
        self.vocab = torchtext.vocab.GloVe(name=embedding_name, dim=embedding_dim, cache='/media/wwk/HDD1/pretrained_models/glove')
        self.vocab.itos.extend(['<unk>'])
        self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
        self.vocab.vectors = torch.cat([self.vocab.vectors, torch.zeros(1, self.vocab.dim)], dim=0)
        self.word_embedding = nn.Embedding.from_pretrained(self.vocab.vectors)

    def generate_label_feats(self, feat, label):
        ori_video_len = feat.shape[0]
        index = np.linspace(start=0, stop=ori_video_len - 1,
                            num=self.segment_num).astype(np.int32)
        new_video = []
        for i in range(len(index) - 1):
            start = index[i]
            end = index[i + 1]
            if start == end or start + 1 == end:
                new_video.append(feat[start])
            else:
                new_video.append(np.mean(feat[start: end], 0))
        new_video.append(feat[-1])
        feat = np.stack(new_video, 0)
        try:
            label[0] = min(np.where(index >= label[0])[0])
        except:
            print(label, index)
        if label[1] == ori_video_len - 1:
            label[1] = self.segment_num - 1
        else:
            label[1] = max(np.where(index <= label[1])[0])
        if label[1] < label[0]:
            label[0] = label[1]
        return feat, label

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        # feat = np.load(os.path.join(self.feat_path, '{}.npy'.format(self.datas[item]['vid'])))
        feat = self.feat[self.datas[item]['vid']][:]

        duration = self.datas[item]['duration']
        timestamp = self.datas[item]['timestamp']


        feat = torch.from_numpy(feat)
        feat = average_to_fixed_length(feat, self.segment_num)

        start_frame = max(self.segment_num * timestamp[0] / duration, 0)
        end_frame = min(self.segment_num * timestamp[1] / duration, self.segment_num-1)
        if start_frame > end_frame:
            start_frame = end_frame
        label = np.asarray([start_frame, end_frame]).astype(np.int32)

        word_idxs = torch.tensor([self.vocab.stoi.get(w, len(self.vocab.stoi)-1) \
        for w in self.datas[item]['words'].strip().split()], dtype=torch.long)
        embedding = self.word_embedding(word_idxs)
        embedding_length = embedding.shape[0]

        if embedding_length > self.max_embedding_length:
            embedding_padded = embedding[: self.max_embedding_length, :]
            embedding_length = self.max_embedding_length
        else:
            embedding_padded = torch.zeros(
                (self.max_embedding_length, embedding.shape[1]))
            embedding_padded[: embedding.shape[0], :] = embedding

        scores, scores_mask, adj_mat = generate_scores(
            self.proposals, label, self.segment_num, self.config['thres_score'], self.config['thres_adjmat'])

        score_nm = []
        for i in range(self.segment_num):
            if i >= label[0] and i <= label[1]:
                score_nm.append(1)
            else:
                score_nm.append(0)
        score_nm = torch.tensor(score_nm).float()

        return {
            'embedding': embedding_padded,
            'feat': feat,
            'embedding_length': embedding_length,
            'label': label,
            'duration': duration,
            'vid': self.datas[item]['vid'],
            'score': scores,
            'score_nm': score_nm,
            'score_mask': scores_mask,
            'proposals': self.proposals.astype(np.float32),
            'adj_mat': adj_mat,
            'index': self.datas[item]['index']
        }
