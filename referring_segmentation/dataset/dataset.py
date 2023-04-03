import os
from PIL import Image
from torch.utils import data
import numpy as np
import h5py
from torchvision import transforms
from dataset import augmentation
import torch
from tqdm import tqdm
from utils.video_reader import clip_annotation_reader, sequence_reader
import json
import torchtext
import torch.nn as nn


class MyDataset(data.Dataset):
    def __init__(self, config, mode='train'):
        super(MyDataset, self).__init__()
        self.input_size = config['input_size']
        self.clip_size = config['clip_size']
        self.datasets = config['{}ing_datasets'.format(mode)]
        self.dataset_root = config['datasets_root']
        self.max_embedding_length = config['max_embedding_length']
        self.mode = mode
        if type(self.datasets) != list:
            self.datasets = [self.datasets]
        print('Preparing datasets: {}'.format(self.datasets))
        self.datas = []
        augmen = [augmentation.FixedResize(self.input_size)]
        if mode == 'train':
            if config['augmentations']['random_crop']:
                augmen.append(augmentation.RandomScale((1.0, 1.1)))
                augmen.append(augmentation.ExtRandomCrop(self.input_size, pad_if_needed=True))
            if config['augmentations']['random_flip']:
                augmen.append(augmentation.RandomHorizontalFlip())
        augmen.append(augmentation.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        augmen.append(augmentation.ToTensor())
        self.transformation = transforms.Compose(augmen)
    
        for dataset in self.datasets:
            assert os.path.exists('./data/{}_{}.json'.format(dataset.lower(), mode)), 'json file not exist: {}'.format('./data/{}_{}.json'.format(dataset.lower(), mode))
            with open('./data/{}_{}.json'.format(dataset.lower(), mode), 'r') as f:
                videosets = json.load(f)

            for video_file, attribute in tqdm(videosets.items()):
                video_root, annotation_root, instances = attribute['frames'], attribute['labels'], attribute['instances']
                if mode == 'train':
                    video_data = clip_annotation_reader(os.path.join(self.dataset_root, video_root), os.path.join(self.dataset_root, annotation_root), \
                                                    instances, self.clip_size, annotation_center=False, dataset=dataset)
                else:
                    video_data = sequence_reader(os.path.join(self.dataset_root, video_root), os.path.join(self.dataset_root, annotation_root), instances, dataset=dataset)
                self.datas += video_data
        
        embedding_name, embedding_dim = config['embedding_type'].split(
            '_')[1], int(config['embedding_type'].split('_')[2])
        self.vocab = torchtext.vocab.GloVe(
            name=embedding_name, dim=embedding_dim)
        self.vocab.itos.extend(['<unk>'])
        self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
        self.vocab.vectors = torch.cat(
            [self.vocab.vectors, torch.zeros(1, self.vocab.dim)], dim=0)
        self.word_embedding = nn.Embedding.from_pretrained(self.vocab.vectors)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        frames = []
        annotations = []
        is_annotated = []
        frame_names = []
        instance = self.datas[item]['instance']

        word_idxs = torch.tensor([self.vocab.stoi.get(w, len(self.vocab.stoi)-1)
                                  for w in self.datas[item]['sentence'].strip().split()], dtype=torch.long)
        embedding = self.word_embedding(word_idxs)
        embedding_length = embedding.shape[0]
        if embedding_length > self.max_embedding_length:
            embedding_padded = embedding[: self.max_embedding_length, :]
            embedding_length = self.max_embedding_length
        else:
            embedding_padded = torch.zeros(
                (self.max_embedding_length, embedding.shape[1]))
            embedding_padded[: embedding.shape[0], :] = embedding

        for i in range(len(self.datas[item]['frames'])):
            frame_names.append(self.datas[item]['frames'][i].split('/')[-1].split('.')[0])
            frame = Image.open(self.datas[item]['frames'][i]).convert('RGB')
            frames.append(frame)
            w, h = frame.size

            sign = True
            if self.datas[item]['label'][i] != 'None':
                with h5py.File(self.datas[item]['label'][i], 'r') as file_annotation:
                    if int(instance) not in list(file_annotation['instance']):
                        annotation = Image.new('L', (w, h))
                    else:
                        if len(file_annotation['reMask'].shape) != 3:
                            annotation = file_annotation['reMask'][:]
                        else:
                            annotation = file_annotation['reMask'][np.where(file_annotation['instance'][:] == int(instance))][0]
                        annotation = Image.fromarray(annotation.T)
            else:
                annotation = Image.new('L', (w, h))
                sign = False
            annotations.append(annotation)
            is_annotated.append(sign)

        sample = {}
        sample['frames'] = frames
        sample['label'] = annotations
        sample = self.transformation(sample)
        sample['word_embedding'] = embedding_padded
        sample['embedding_length'] = embedding_length
        sample['is_annotated'] = is_annotated
        sample['video'] = self.datas[item]['video']
        sample['name'] = frame_names
        sample['dataset'] = self.datas[item]['dataset']
        sample['instance'] = instance

        return sample





