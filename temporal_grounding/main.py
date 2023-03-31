import argparse
import os
import json
import numpy as np
import random
import torch
from dataset import MyDataset
from model.model import Model
import torch.nn as nn
from utils.trainer import Trainer
from utils.tester import Tester


def main(args):
    with open(args.json_file, 'r') as f:
        config = json.load(f)
    config['mode'] = args.mode
    if args.checkpoint is not None:
        config['checkpoint'] = args.checkpoint

    is_cuda = config['cuda']
    if is_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # init_distributed_mode()
    if config['mode'] == 'train':
        dataset = MyDataset(config, 'train')
        eval_dataset = MyDataset(config, 'test')
    elif config['mode'] == 'test':
        dataset = MyDataset(config, 'test')
    else:
        raise NotImplementedError

    model = Model(config)
    if is_cuda:
        model = nn.DataParallel(model)
        model = model.cuda()

    if not os.path.exists(config['log_root']):
        os.makedirs(config['log_root'])
    if config['mode'] == 'train':
        trainer = Trainer(config)
        trainer.train(model, dataset, eval_dataset)
    elif config['mode'] == 'test':
        tester = Tester(config)
        tester.test(model, dataset)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # python main.py --json_file=json/config_Charades-STA_I3D_regression.json --mode=train
    # python main.py --json_file=json/config_Charades-STA_I3D_anchor.json --mode=train
    # python main.py --json_file=json/config_ActivityNet_C3D_regression.json --mode=train
    # python main.py --json_file=json/config_ActivityNet_C3D_anchor.json --mode=train
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str,
                        default='json/config_Charades-STA_I3D_regression.json', required=True)
    parser.add_argument('--mode', type=str,
                        default='train', required=True, choices=['train', 'test'])
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    main(args)
