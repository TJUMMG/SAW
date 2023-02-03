from data.generate_data_list import generate_data_list
from data.generate_word_embedding import generate_word_embedding_h5
import os


def check_datas(datasets=['A2D','JHMDB'], root='/media/HardDisk/wwk/video_text/datasets/', embedding_type='bert'):

    data_list_root = './data/data_list'
    word_embedding_root = './data/word_embeddings'
    completion = True

    print('------------Checking word embedding-----------')
    embedddings = ['data_txt_{}_{}.h5'.format(dataset.lower(), embedding_type) for dataset in datasets]
    if not os.path.exists(word_embedding_root):
        completion = False
        print('Missing word embeddings: {}'.format(embedddings))
        print('Preparing for word embeddings....')
        os.mkdir(word_embedding_root)
        for dataset in datasets:
            generate_word_embedding_h5(dataset=dataset, embedding_type=embedding_type, save_root=word_embedding_root)
    else:
        embedding_files = os.listdir(word_embedding_root)
        if len(embedding_files) == 0:
            print('Missing word embeddings: {}'.format(embedddings))
            completion = False
            for dataset in datasets:
                generate_word_embedding_h5(dataset=dataset, embedding_type=embedding_type, save_root=word_embedding_root)
        else:
            sign = True
            for embeddding in embedddings:
                if embeddding not in embedding_files:
                    print('Missing data lists: {}'.format(embeddding))
                    completion = False
                    sign = False
            if not sign:
                for dataset in datasets:
                    generate_word_embedding_h5(dataset=dataset, embedding_type=embedding_type, save_root=word_embedding_root)
            else:
                print('Word embedding files complete.')

    data_txts = ['data_a2d_train.json', 'data_a2d_test.json', 'data_jhmdb_test.json']
    print('------------Checking data list-----------')
    if not os.path.exists(data_list_root):
        print('Missing data lists: {}'.format(data_txts))
        print('Preparing for data list....')
        os.mkdir(data_list_root)
        generate_data_list(datasets, root, data_list_root)
        completion = False
    else:
        jsons = [f for f in os.listdir(data_list_root) if 'json' in f]
        if len(jsons) == 0:
            print('Missing data lists: {}'.format(data_txts))
            completion = False
            generate_data_list(datasets, root, data_list_root)
        else:
            sign = True
            for txt in data_txts:
                if txt not in jsons:
                    print('Missing data lists: {}'.format(txt))
                    completion = False
                    sign = False
            if not sign:
                generate_data_list(datasets, root, data_list_root)
            else:
                print('Data_list files complete.')

    if completion:
        return True
    else:
        return False