from __future__ import print_function

import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms

import models


class MainSolver(object):
    def __init__(self, args):
        self.args = args
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.history_score = np.zeros((args.epochs - args.start_epoch + 1, 3))

    def initialize_dataset(self):
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}

        if self.args.dataset == 'cifar10':
            self.train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data.cifar10', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.Pad(4),
                                     transforms.RandomCrop(32),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                 ])),
                batch_size=self.args.batch_size, shuffle=True, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])),
                batch_size=self.args.test_batch_size, shuffle=True, **kwargs)

        else:
            self.train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('./data.cifar100', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.Pad(4),
                                      transforms.RandomCrop(32),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                  ])),
                batch_size=self.args.batch_size, shuffle=True, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])),
                batch_size=self.args.test_batch_size, shuffle=True, **kwargs)

    def initialize_model(self):
        self.model = models.__dict__[self.args.arch](dataset=self.args.dataset, depth=self.args.depth).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

    def initialize_all(self):
        torch.manual_seed(self.args.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.args.seed)

        if not os.path.exists(self.args.save):
            os.makedirs(self.args.save)

        self.initialize_dataset()
        self.initialize_model()

    def update_bn(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(self.args.s * torch.sign(m.weight.data))

    def train(self, epoch):
        self.model.train()
        avg_loss = 0.
        train_acc = 0.
        for batch_i, (_d, _t) in enumerate(self.train_loader):
            _d, _t = _d.to(self.device), _t.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(_d)
            loss = F.cross_entropy(output, _t)
            avg_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            train_acc += pred.eq(_t.data.view_as(pred)).cpu().sum()
            loss.backward()
            if self.args.sr:
                self.update_bn()
            self.optimizer.step()
            if batch_i % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_i * len(_d), len(self.train_loader.dataset), 100. * batch_i / len(self.train_loader),
                    loss.item()))
        self.history_score[epoch][0] = avg_loss / len(self.train_loader)
        self.history_score[epoch][1] = train_acc / float(len(self.train_loader))

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for _d, _t in self.test_loader:
                _d, _t = _d.to(self.device), _t.to(self.device)
                output = self.model(_d)
                test_loss += F.cross_entropy(output, _t, size_average=False).item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(_t.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        return correct / float(len(self.test_loader.dataset))

    @staticmethod
    def save_checkpoint(state, is_best, filepath):
        torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

    def run(self):
        self.initialize_all()

        best_prec = 0.
        for epoch in range(self.args.start_epoch, self.args.epochs):
            if epoch in [self.args.epochs * 0.5, self.args.epochs * 0.75]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
            self.train(epoch)
            prec = self.test()
            self.history_score[epoch][2] = prec
            np.savetxt(os.path.join(self.args.save, 'record.txt'), self.history_score, fmt='%10.5f', delimiter=',')
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_prec1': best_prec,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, filepath=self.args.save)

        print("Best accuracy: " + str(best_prec))
        self.history_score[-1][0] = best_prec
        np.savetxt(os.path.join(self.args.save, 'record.txt'), self.history_score, fmt='%10.5f', delimiter=',')