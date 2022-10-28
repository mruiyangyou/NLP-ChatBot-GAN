from solver import Solver
from data_loder import get_loader
from Config import DefautConfig

import torch
import numpy
import argparse
import os

config = DefautConfig()

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    svhn_loader, mnist_loader = get_loader(config)

    solver = Solver(config, svhn_loader, mnist_loader)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    if config.mode == 'train':
        solver.train()

    elif config.mode == 'sample':
        solver.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoches', type = int, default=40000)
    parser = parser.parse_args()
    config.train_iter = parser.epoches
    main(config)




