import argparse
import os
import sys


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')#cifar10 mnist
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--penalty_anneal_iters', type=int, default=50)
parser.add_argument('--penalty_weight', type=float, default=10.0)
parser.add_argument('--d', type=int, default= 0)
parser.add_argument('--resume', default=False)
parser.add_argument('--mode', default='IRM')#IRM/STD
args = parser.parse_args()


class Configuration(object):
    lr = 0.001
    adam = True
    val_interval=5
    weight_decay = 0.0
    batch_size=64
    steps=501
    total_epochs = 100
    milestones = [50, 75]
    attack_penalty=[1.,5.]



config = Configuration()