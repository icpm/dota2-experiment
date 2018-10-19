from __future__ import print_function

import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import dataloader
from misc import progress_bar, record_info
from models.model import get_model, get_zero_short_model


class OneHot(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.pretrained = config.pretrained
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = None
        self.seed = config.seed
        self.train_loader = None
        self.test_loader = None

    def build_model(self):
        if self.GPU_IN_USE:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.model = get_zero_short_model(pretrained=self.pretrained).to(self.device)
        self.model.load_state_dict(torch.load('./models/model.pth'))
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)

    def build_dataloader(self):
        self.train_loader, self.test_loader = dataloader.get_dataloader(self.batch_size, self.test_batch_size)

    def test(self):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device).float()
                prediction = self.model(data)

                # print(prediction.shape)
                total += data.size(0)
                progress_bar(batch_num, len(self.test_loader), 'test loss: %.4f | accuracy: %.4f'
                             % (test_loss / (batch_num + 1), test_correct / total))
        return test_loss / total, test_correct / total

    def save_data(self, s_a, s_l):
        result_dir = './ont_hot_pretrained' if self.pretrained else './ont_hot'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        test_acc = {'Test Accuracy': [s_a]}
        test_loss = {'Test Loss': [s_l]}
        record_info(test_acc, result_dir + '/test_acc.csv')
        record_info(test_loss, result_dir + '/test_loss.csv')

    def run(self):
        self.build_model()
        self.build_dataloader()
        for epoch in range(1, self.epochs + 1):
            fold = (epoch - 1) % 5
            print("\n===> Epoch {} starts: (fold: {})".format(epoch, fold))

            test_loss, test_accuracy = self.test()
            # self.save_data(test_accuracy, test_loss)


class MultiHot(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.pretrained = config.pretrained
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = None
        self.seed = config.seed
        self.train_loader = None
        self.test_loader = None

    def build_model(self):
        if self.GPU_IN_USE:
            cudnn.benchmark = True
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = get_model(pretrained=self.pretrained).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)

    def build_dataloader(self, fold):
        self.train_loader, self.test_loader = dataloader.get_dataloader(self.batch_size, self.test_batch_size, fold)

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device).float()
            self.optimizer.zero_grad()
            prediction = self.model(data)
            loss = self.criterion(prediction, target)
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step(loss.data.cpu().numpy())
            train_loss += loss.item()
            train_correct += np.sum(torch.max(prediction, 1)[1].cpu().numpy() == torch.max(target, 1)[1].cpu().numpy())
            total += data.size(0)
            progress_bar(batch_num, len(self.train_loader), 'train loss: %.4f | accuracy: %.4f'
                         % (train_loss / (batch_num + 1), train_correct / total))
        return train_loss / total, train_correct / total

    def test(self):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device).float()
                prediction = self.model(data)
                loss = self.criterion(prediction, target)
                test_loss += loss.item()
                test_correct += np.sum(torch.max(prediction, 1)[1].cpu().numpy() == torch.max(target, 1)[1].cpu().numpy())
                total += data.size(0)
                progress_bar(batch_num, len(self.test_loader), 'test loss: %.4f | accuracy: %.4f'
                             % (test_loss / (batch_num + 1), test_correct / total))
        return test_loss / total, test_correct / total

    def save_data(self, t_a, t_l, s_a, s_l):
        result_dir = './multi_hot_pretrained' if self.pretrained else './multi_hot'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        train_acc = {'Train Accuracy': [t_a]}
        train_loss = {'Train Loss': [t_l]}
        test_acc = {'Test Accuracy': [s_a]}
        test_loss = {'Test Loss': [s_l]}
        record_info(train_acc, result_dir + '/train_acc.csv')
        record_info(train_loss, result_dir + '/train_loss.csv')
        record_info(test_acc, result_dir + '/test_acc.csv')
        record_info(test_loss, result_dir + '/test_loss.csv')

    def run(self):
        self.build_model()
        for epoch in range(1, self.epochs + 1):
            fold = (epoch - 1) % 5
            print("\n===> Epoch {} starts: (fold: {})".format(epoch, fold))
            self.build_dataloader(fold)

            train_loss, train_accuracy = self.train()
            test_loss, test_accuracy = self.test()

            self.save_data(train_accuracy, train_loss, test_accuracy, test_loss)
