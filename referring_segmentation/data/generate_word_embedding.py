from flair.embeddings import WordEmbeddings, BertEmbeddings, OneHotEmbeddings
from flair.data import Sentence
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm


def generate_word_embedding_h5(dataset='A2D', root='/media/HardDisk/wwk/video_text/datasets/', embedding_type='bert', save_root = './data/word_embedding'):
    if embedding_type == 'glove100d':
        embedding_tool = WordEmbeddings('glove')
    elif embedding_type == 'glove300d':
        embedding_tool = load_glove_feats('./model/pretrained/glove.840B.300d.txt')
    elif embedding_type == 'bert':
        embedding_tool = BertEmbeddings()

    txt_root = os.path.join(root, dataset, '{}_annotation.txt'.format(dataset.lower()))
    num_data = 0
    with open(txt_root, 'r') as f:
        lines = f.readlines()
        name_previous = None
        with h5py.File(os.path.join(save_root, 'data_txt_{}_{}.h5'.format(dataset.lower(), embedding_type)), 'a') as hf:
            for i, line in enumerate(lines):
                line = line.strip()
                if i > 0:
                    if dataset == 'A2D':
                        name, instance, descr = line.split(',')
                    else:
                        name, descr = line.split(',')
                        instance = '0'
                    print('[{}\{}] processing.... Video num: {} Video name: {}'.format(i, len(lines), num_data, name))
                    if embedding_type == 'glove100d' or embedding_type == 'bert':
                        sentence = Sentence(descr)
                        embedding_tool.embed(sentence)
                        embedding = []
                        for token in sentence:
                            embedding.append(token.embedding.unsqueeze(0))
                        embedding = torch.cat(embedding, dim=0).data.cpu().numpy()
                    else:
                        sentence = Sentence(descr)
                        embedding = []
                        for token in sentence:
                            embedding.append(np.expand_dims(np.array(embedding_tool[token.text]), axis=0))
                        embedding = np.concatenate(embedding, axis=0)
                    if name != name_previous:
                        # n_des = 0
                        num_data += 1
                        hf_video = hf.create_group(name)
                        # if dataset == 'a2d':
                        #     hf_instance = hf_video.create_group('instance')
                        hf_video.create_dataset(instance, data=embedding)
                        name_previous = name
                    #
                    else:
                        hf_video = hf[name]
                        hf_video.create_dataset(instance, data=embedding)
                        name_previous = name
            hf.close()
        f.close()


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