import os
from PIL import Image
from torch.utils import data
import numpy as np
import h5py
from torchvision import transforms
from data import augmentation
import torch
from tqdm import tqdm
from utils.video_reader import clip_annotation_reader, sequence_reader, clip_sequence_reader, single_frame_reader, single_frame_with_text_reader
import scipy.io as scio
import json


class MyDatasetTrain(data.Dataset):
    def __init__(self, config):
        super(MyDatasetTrain, self).__init__()

        self.input_size = config['input_size']
        self.clip_size = config['clip_size']
        self.datasets = config['training_datasets']
        self.embedding_type = config['embedding_type']
        self.max_embedding_length = config['max_embedding_length']
        if type(self.datasets) != list:
            self.datasets = [self.datasets]
        print('Preparing datasets: {}'.format(self.datasets))
        self.datas = []
        augmen = [augmentation.FixedResize(self.input_size)]
        if config['augmentations']['random_crop']:
            augmen.append(augmentation.RandomScale((1.0, 1.1)))
            augmen.append(augmentation.ExtRandomCrop(self.input_size, pad_if_needed=True))
        if config['augmentations']['random_flip']:
            augmen.append(augmentation.RandomHorizontalFlip())
        augmen.append(augmentation.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        augmen.append(augmentation.ToTensor())
        self.transformation = transforms.Compose(augmen)
    
        for dataset in self.datasets:
            assert dataset == 'A2D' or dataset == 'JHMDB', ('incorrect dataset: {}'.format(dataset))
            with open('./data/data_list/data_{}_train.json'.format(dataset.lower()), 'r') as f:
                videosets = json.load(f)

            for video_file, attribute in tqdm(videosets.items()):
                # video_root, annotation_root, instance_num = video_file.split(' ')
                video_root, annotation_root, instances = attribute['frames'], attribute['labels'], attribute['instances']

                video_data = clip_annotation_reader(video_root, annotation_root, instances, self.clip_size, annotation_center=False, dataset=dataset)
                self.datas += video_data

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        frames = []
        annotations = []
        is_annotated = []
        instance = self.datas[item]['instance']

        with h5py.File('./data/word_embeddings/data_txt_{}_{}.h5'.format(self.datas[item]['dataset'].lower(), self.embedding_type), 'r') as embedding_file:
            try:
                word_embedding = embedding_file[self.datas[item]['video']][str(instance)]
            except KeyError:
                print(self.datas[item]['video'])


            word_embedding_padded = np.zeros((self.max_embedding_length, word_embedding.shape[1]))
            word_embedding_padded[:word_embedding.shape[0], :] = word_embedding
            embedding_length = word_embedding.shape[0]

        for i in range(self.clip_size):
            frame = Image.open(self.datas[item]['frames'][i]).convert('RGB')
            frames.append(frame)
            w, h = frame.size

            if self.datas[item]['label'][i] != 'None':
                with h5py.File(self.datas[item]['label'][i], 'r') as file_annotation:
                    # print(file_annotation['reMask'].shape)
                    if instance not in list(file_annotation['instance']):
                        annotation = Image.new('L', (w, h))
                        sign = 1
                    else:
                        if len(file_annotation['reMask'].shape) != 3:
                            annotation = file_annotation['reMask'][:]
                        else:
                            annotation = file_annotation['reMask'][np.where(file_annotation['instance'][:] == instance)][0]
                        annotation = Image.fromarray(annotation.T)
                        sign = 1
            else:
                annotation = Image.new('L', (w, h))
                sign = 0
            annotations.append(annotation)
            is_annotated.append(sign)

        sample = {}
        sample['frames'] = frames
        sample['label'] = annotations
        sample = self.transformation(sample)
        sample['word_embedding'] = torch.from_numpy(word_embedding_padded).float()
        sample['embedding_length'] = embedding_length
        sample['is_annotated'] = is_annotated

        return sample


class MyDatasetTest(data.Dataset):
    def __init__(self, config):
        super(MyDatasetTest, self).__init__()
        self.input_size = config['input_size']
        self.datasets = config['testing_datasets']
        self.embedding_type = config['embedding_type']
        self.max_embedding_length = config['max_embedding_length']
        if type(self.datasets) != list:
            self.datasets = [self.datasets]
        print('Preparing datasets: {}'.format(self.datasets))
        self.datas = []
        self.transformation = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        ])

        for dataset in self.datasets:
            assert dataset == 'A2D' or dataset == 'JHMDB', ('incorrect dataset: {}'.format(dataset))

            with open('./data/data_list/data_{}_test.json'.format(dataset.lower()), 'r') as f:
                videosets = json.load(f)

            for video_file, attribute in tqdm(videosets.items()):
                video_root, annotation_root, instances = attribute['frames'], attribute['labels'], attribute['instances']

                if config['testing_read_mode'] == 'sequence':
                    video_data = sequence_reader(video_root, annotation_root, instances, dataset=dataset)
                elif config['testing_read_mode'] == 'clip':
                    video_data = clip_sequence_reader(video_root, annotation_root, instances, config['clip_size'], config['testing_read_step'], dataset)
                else:
                    raise NotImplementedError
                self.datas += video_data

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        frames = []
        annotations = []
        is_annotated = []
        name = []
        instance = self.datas[item]['instance']

        with h5py.File('./data/word_embeddings/data_txt_{}_{}.h5'.format(self.datas[item]['dataset'].lower(),
                                                                         self.embedding_type), 'r') as embedding_file:
            word_embedding = embedding_file[self.datas[item]['video']][str(instance)]

            word_embedding_padded = np.zeros((self.max_embedding_length, word_embedding.shape[1]))
            word_embedding_padded[:word_embedding.shape[0], :] = word_embedding
            embedding_length = word_embedding.shape[0]

        for i in range(len(self.datas[item]['frames'])):
            name.append(self.datas[item]['frames'][i].split('/')[-1].split('.')[0])
            frame = Image.open(self.datas[item]['frames'][i]).convert('RGB')
            w, h = frame.size
            frame = self.transformation(frame)
            frames.append(frame)

            if self.datas[item]['label'][i] != 'None':
                if self.datas[item]['dataset'] == 'A2D':
                    with h5py.File(self.datas[item]['label'][i], 'r') as file_annotation:
                        # print(file_annotation['reMask'].shape)
                        if len(file_annotation['reMask'].shape) != 3:
                            annotation = file_annotation['reMask'][:]
                        else:
                            annotation = file_annotation['reMask'][np.where(file_annotation['instance'][:] == instance)][0]
                        annotation = torch.from_numpy(annotation.T).unsqueeze(0)
                        sign = 1
                else:

                    annotation = scio.loadmat(self.datas[item]['label'][i])
                    # try:
                    annotation = annotation['part_mask'][:, :, i]
                    # except IndexError:
                    #     print(self.datas[item]['video'])
                    annotation = torch.from_numpy(annotation).unsqueeze(0)
                    sign = 1

            else:
                annotation = torch.zeros((1, h, w))
                sign = 0
            annotations.append(annotation)
            is_annotated.append(sign)

        sample = {}
        sample['frames'] = frames
        sample['label'] = annotations
        sample['word_embedding'] = torch.from_numpy(word_embedding_padded).float()
        sample['embedding_length'] = embedding_length
        sample['is_annotated'] = is_annotated
        sample['video'] = self.datas[item]['video']
        sample['name'] = name
        sample['dataset'] = self.datas[item]['dataset']
        sample['instance'] = instance

        return sample


class MyDatasetTrainBackbone(data.Dataset):
    def __init__(self, config):
        super(MyDatasetTrainBackbone, self).__init__()

        self.input_size = config['input_size']
        self.clip_size = config['clip_size']
        self.datasets = config['training_datasets']
        self.embedding_type = config['embedding_type']
        self.max_embedding_length = config['max_embedding_length']
        if type(self.datasets) != list:
            self.datasets = [self.datasets]
        print('Preparing datasets: {}'.format(self.datasets))
        self.datas = []
        augmen = [augmentation.FixedResize(self.input_size)]
        if config['augmentations']['random_crop']:
            augmen.append(augmentation.RandomScale((1.0, 1.1)))
            augmen.append(augmentation.ExtRandomCrop(self.input_size, pad_if_needed=True))
        if config['augmentations']['random_flip']:
            augmen.append(augmentation.RandomHorizontalFlip())
        augmen.append(augmentation.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        augmen.append(augmentation.ToTensor())
        self.transformation = transforms.Compose(augmen)

        for dataset in self.datasets:
            assert dataset == 'A2D' or dataset == 'JHMDB', ('incorrect dataset: {}'.format(dataset))
            with open('./data/data_list/data_{}_train.json'.format(dataset.lower()), 'r') as f:
                videosets = json.load(f)

            for video_file, attribute in tqdm(videosets.items()):
                # video_root, annotation_root, instance_num = video_file.split(' ')
                video_root, annotation_root, instances = attribute['frames'], attribute['labels'], attribute[
                    'instances']
                # video_data = single_frame_with_text_reader(video_root, annotation_root, instances)
                video_data = single_frame_reader(video_root, annotation_root, dataset)
                self.datas += video_data
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):

        name = self.datas[item]['frame'].split('/')[-1].split('.')[0]
        # instance = self.datas[item]['instance']
        frame = Image.open(self.datas[item]['frame']).convert('RGB')
        w, h = frame.size

        # with h5py.File('./data/word_embeddings/data_txt_{}_{}.h5'.format(self.datas[item]['dataset'].lower(), self.embedding_type), 'r') as embedding_file:
        #     try:
        #         word_embedding = embedding_file[self.datas[item]['video']][str(instance)]
        #     except KeyError:
        #         print(self.datas[item]['video'])
        #
        #
        #     word_embedding_padded = np.zeros((self.max_embedding_length, word_embedding.shape[1]))
        #     word_embedding_padded[:word_embedding.shape[0], :] = word_embedding
        #     embedding_length = word_embedding.shape[0]


        with h5py.File(self.datas[item]['label'], 'r') as file_annotation:
            # print(file_annotation['reMask'].shape)
            if len(file_annotation['reMask'].shape) != 3:
                annotation = file_annotation['reMask'][:]
            else:
            #     annotation = file_annotation['reMask'][np.where(file_annotation['instance'][:] == instance)][0]
                annotation = np.zeros((w, h))
                for instance in range(file_annotation['reMask'].shape[0]):
                    annotation[np.where(file_annotation['reMask'][instance]>0)] = 1
            annotation = Image.fromarray(annotation.T)


        sample = {}
        sample['frames'] = [frame]
        sample['label'] = [annotation]
        sample = self.transformation(sample)
        sample['frame'] = sample['frames'][0]
        sample['label'] = sample['label'][0]
        sample['video'] = self.datas[item]['video']
        sample['name'] = name
        sample['dataset'] = self.datas[item]['dataset']

        return sample







