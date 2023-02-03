import argparse
import os
import json
import numpy as np
import random
import torch
from data.dataset import MyDatasetTrain, MyDatasetTest, MyDatasetTrainBackbone
from model.model import Model
import torch.nn as nn
from utils.trainer import Trainer
from utils.tester import Tester


def main(args):
    with open(args.json_file, 'r') as f:
        config = json.load(f)

    setting = config['setting_config']
    data_config = config['data_config']
    model_config = config['model_config']
    training_config = config['training_config']
    testing_config = config['testing_config']
    is_cuda = setting['cuda']
    if is_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = setting['gpu_id']

    torch.manual_seed(setting['seed'])
    np.random.seed(setting['seed'])
    random.seed(setting['seed'])

    training_config['cuda'] = is_cuda
    testing_config['cuda'] = is_cuda
    training_config['train_backbone'] = model_config['train_backbone']

    # print('------------Checking data integrity-----------')
    # from utils.check_datas import check_datas
    # is_file_completion = check_datas(['A2D', 'JHMDB'], data_config['datasets_root'], data_config['embedding_type'])
    # if not is_file_completion:
    #     print('------------Checking data integrity again-----------')
    #     is_file_completion = check_datas(['A2D', 'JHMDB'], data_config['datasets_root'], data_config['embedding_type'])
    # assert is_file_completion, ('Some problem occurs in loading data')

    # init_distributed_mode()
    if setting['mode'] == 'train' and not model_config['train_backbone']:
        dataset = MyDatasetTrain(data_config)
        val_dataset = MyDatasetTest(data_config)
    elif setting['mode'] == 'test':
        dataset = MyDatasetTest(data_config)
    elif setting['mode'] == 'train' and model_config['train_backbone']:
        dataset = MyDatasetTrainBackbone(data_config)
    else:
        raise NotImplementedError

    if model_config['train_backbone']:
        # model = Backbone(model_config)
        pass
    else:
        model = Model(model_config)
    if is_cuda:
        model = nn.DataParallel(model)
        model = model.cuda()
    
    if not os.path.exists(training_config['log_root']):
        os.mkdir(training_config['log_root'])
    if setting['mode'] == 'train':
        trainer = Trainer(training_config)
        trainer.train(model, dataset, val_dataset)
    elif setting['mode'] == 'test':
        tester = Tester(testing_config)
        tester.test(model, dataset)
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='json/config.json')
    args = parser.parse_args()
    main(args)




