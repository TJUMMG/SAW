import argparse
import os
import json
import numpy as np
import random
import torch
from dataset.dataset import MyDataset
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
    if args.checkpoint is not None:
        testing_config['checkpoint'] = args.checkpoint

    torch.manual_seed(setting['seed'])
    np.random.seed(setting['seed'])
    random.seed(setting['seed'])

    training_config['cuda'] = is_cuda
    testing_config['cuda'] = is_cuda
    training_config['train_backbone'] = model_config['train_backbone']

    # init_distributed_mode()
    if args.mode == 'train':
        dataset = MyDataset(data_config, 'train')
        val_dataset = MyDataset(data_config, 'test')
    elif args.mode == 'test':
        dataset = MyDataset(data_config, 'test')
    else:
        raise NotImplementedError

    model = Model(model_config)
    if is_cuda:
        model = nn.DataParallel(model)
        model = model.cuda()

    if not os.path.exists(training_config['log_root']):
        os.mkdir(training_config['log_root'])
    if args.mode == 'train':
        trainer = Trainer(training_config)
        trainer.train(model, dataset, val_dataset)
    elif args.mode == 'test':
        tester = Tester(testing_config)
        tester.test(model, dataset)
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='json/config_a2d_sentences.json')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    main(args)
