from __future__ import print_function
import argparse
from solver import *

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=256, help='testing batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=2e-2, help='Learning Rate. Default=0.01')
parser.add_argument('--pretrained', '-p', action='store_true', help='whether to use pretrained model')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')


args = parser.parse_args()


def main():
    domain_adaption_solver = OneHot(args)
    domain_adaption_solver.run()


if __name__ == '__main__':
    main()
