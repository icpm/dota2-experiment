import argparse
import torch
from solver import VGGPrune


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar100', help='training dataset (default: cifar10)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--depth', type=int, default=19, help='depth of the vgg')
    parser.add_argument('--percent', type=float, default=0.5, help='scale sparse rate (default: 0.5)')
    parser.add_argument('--model', default='', type=str, metavar='PATH', help='path to the model (default: none)')
    parser.add_argument('--save', default='', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    solver = VGGPrune(args)
    solver.run()
