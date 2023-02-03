from flair.embeddings import WordEmbeddings, BertEmbeddings
from flair.data import Sentence
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import json


def generate_word_embedding_h5(dataset='TACoS', root='/media/HardDisk/wwk/video_text/datasets/TACoS/text/index.tsv', embedding_type='glove300d', save_root='/media/HardDisk/wwk/video_text/datasets/TACoS/embeddings'):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if embedding_type == 'glove100d':
        embedding_tool = WordEmbeddings('glove')
    elif embedding_type == 'glove300d':
        embedding_tool = load_glove_feats(
            '/media/HardDisk/wwk/video_text/pretrained/glove.840B.300d.txt')
    elif embedding_type == 'word2vec':
        embedding_tool = WordEmbeddings(
            '/media/HardDisk/wwk/video_text/pretrained/GoogleNews-vectors-negative300.bin')
    elif embedding_type == 'bert':
        embedding_tool = BertEmbeddings()
    datas = {}
    if dataset == 'TACoS':
        datas = {}
        train_infos = json.load(open(os.path.join(root, 'train.json')))
        val_infos = json.load(open(os.path.join(root, 'val.json')))
        test_infos = json.load(open(os.path.join(root, 'test.json')))
        for key, value in train_infos.items():
            datas[key] = {}
            for i in range(len(value['timestamps'])):
                datas[key][str(i)] = value['sentences'][i]

        for key, value in val_infos.items():
            datas[key] = {}
            for i in range(len(value['timestamps'])):
                datas[key][str(i)] = value['sentences'][i]

        for key, value in test_infos.items():
            datas[key] = {}
            for i in range(len(value['timestamps'])):
                datas[key][str(i)] = value['sentences'][i]

    elif dataset == 'Charades-STA':
        data_train = os.path.join(root, 'charades_sta_train.txt')
        data_test = os.path.join(root, 'charades_sta_test.txt')
        for file in [data_train, data_test]:
            with open(file, 'r') as f:
                data = f.readlines()
            for line in data:
                line = line.strip()
                name = line.split(' ')[0]
                text = line.split('##')[-1]
                if name not in datas.keys():
                    datas[name] = {}
                    n_num = 0
                    datas[name][str(n_num)] = text
                else:
                    n_num += 1
                    datas[name][str(n_num)] = text
    elif dataset == 'ActivityNet':
        datas = {}
        train_infos = json.load(open(os.path.join(root, 'train.json')))
        val_1_infos = json.load(open(os.path.join(root, 'val_1.json')))
        val_2_infos = json.load(open(os.path.join(root, 'val_2.json')))
        for key, value in train_infos.items():
            datas[key] = {}
            for i in range(len(value['timestamps'])):
                datas[key][str(i)] = value['sentences'][i]

        for key, value in val_1_infos.items():
            datas[key] = {}
            for i in range(len(value['timestamps'])):
                datas[key][str(i)] = value['sentences'][i]

        for key, value in val_2_infos.items():
            if key not in datas.keys():
                datas[key] = {}
            if len(datas[key].keys()) > 0:
                n = len(datas[key].keys())
            else:
                n = 0
            for i in range(len(value['timestamps'])):
                datas[key][str(i + n)] = value['sentences'][i]

    num_data = 0
    with h5py.File(os.path.join(save_root, 'embeddings_{}.h5'.format(embedding_type)), 'a') as hf:
        for name in datas.keys():
            name_previous = None
            for query in datas[name].keys():
                descr = datas[name][query]
                if embedding_type == 'glove100d' or embedding_type == 'bert' or embedding_type == 'word2vec':
                    sentence = Sentence(descr)
                    embedding_tool.embed(sentence)
                    embedding = []
                    for token in sentence:
                        embedding.append(token.embedding.unsqueeze(0))
                    embedding = torch.cat(
                        embedding, dim=0).data.cpu().numpy()
                else:
                    sentence = Sentence(descr)
                    embedding = []
                    for token in sentence:
                        try:
                            embedding.append(np.expand_dims(
                                np.array(embedding_tool[token.text]), axis=0))
                        except:
                            embedding_tool[token.text] = np.random.random(size=300)
                            embedding.append(np.expand_dims(embedding_tool[token.text], axis=0))
                    embedding = np.concatenate(embedding, axis=0)
                if name != name_previous:
                    num_data += 1
                    hf_video = hf.create_group(name)
                    hf_video.create_dataset(query, data=embedding)
                    name_previous = name
                else:
                    hf_video = hf[name]
                    hf_video.create_dataset(query, data=embedding)
                    name_previous = name
        hf.close()


def load_glove_feats(glove_path):
    print('loading GloVe feature from {}'.format(glove_path))
    glove_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        with tqdm(total=2196017, desc='Loading GloVe', ascii=True) as pbar:
            for line in f:
                tokens = line.split(' ')
                assert len(tokens) == 301
                word = tokens[0]
                vec = list(map(lambda x: float(x), tokens[1:]))
                glove_dict[word] = vec
                pbar.update(1)
    return glove_dict

if __name__ == '__main__':
    # generate_word_embedding_h5(embedding_type='glove300d')
    generate_word_embedding_h5(embedding_type='glove300d', dataset='TACoS', root='/media/HardDisk/wwk/video_text/datasets/TACoS/split/captions/', save_root='/media/HardDisk/wwk/video_text/datasets/TACoS/embeddings')