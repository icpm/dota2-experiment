from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from models.transform import TransformerNet, VGG16
from torch.backends import cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from data import dataloader
from misc import progress_bar, record_info
from models.model import alexnet


class Stylizer(object):
    def __init__(self, style_image):
        # device configuration
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')

        self.style_image = style_image
        self.style_size = None

        self.gram_style = None
        self.vgg = None

        self.transformer = None
        self.optimizer = None
        self.criterion = None
        self.seed = 42

        self.batch_size = 16
        self.lr = 1e-3
        self.content_weight = 1e5
        self.style_weight = 1e10

    def get_style(self):
        # set up style
        style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        style = self.load_image(self.style_image, size=self.style_size)
        style = style_transform(style)
        style = style.repeat(self.batch_size, 1, 1, 1)

        # set up feature extractor
        self.vgg = VGG16(requires_grad=False).to(self.device)

        style_v = style.to(self.device)
        style_v = self.normalize_batch(style_v)
        features_style = self.vgg(style_v)
        self.gram_style = [self.gram_matrix(y) for y in features_style]

    def build_model(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.transformer = TransformerNet().to(self.device)
        self.optimizer = Adam(self.transformer.parameters(), self.lr)
        self.criterion = torch.nn.MSELoss()

        if self.cuda:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

    @staticmethod
    def gram_matrix(y):
        (batches, channel, height, width) = y.size()
        features = y.view(batches, channel, width * height)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (channel * height * width)
        return gram

    @staticmethod
    def normalize_batch(batch):
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch /= 255.0
        batch -= mean
        batch = batch / std
        return batch

    @staticmethod
    def load_image(filename, size=None, scale=None):
        image = Image.open(filename)
        if size is not None:
            image = image.resize((size, size), Image.ANTIALIAS)
        elif scale is not None:
            image = image.resize((int(image.size[0] / scale), int(image.size[1] / scale)), Image.ANTIALIAS)
        return image

    def train(self, data):
        self.transformer.train()

        target = self.transformer(data)
        data = self.normalize_batch(data)
        target = self.normalize_batch(target)

        target_feature = self.vgg(target)
        data_feature = self.vgg(data)
        content_loss = self.content_weight * self.criterion(target_feature.relu2_2, data_feature.relu2_2)

        style_loss = 0.
        for current_target_feature, current_gram_style in zip(target_feature, self.gram_style):
            current_target_feature = self.gram_matrix(current_target_feature)
            style_loss += self.criterion(current_target_feature, current_gram_style[:len(data), :, :])
        style_loss *= self.style_weight

        # do the backpropagation
        total_loss = content_loss + style_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        content_loss.item()
        style_loss.item()

        # print(target.shape)
        return target

    def validate(self):
        self.get_style()
        self.build_model()


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
        self.stylizer = None

    def build_model(self):
        if self.GPU_IN_USE:
            cudnn.benchmark = True
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = alexnet(pretrained=self.pretrained).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)

        # initialize stylizer
        self.stylizer = Stylizer('./data/dota.jpg')
        self.stylizer.build_model()
        self.stylizer.get_style()

    def build_dataloader(self, fold):
        self.train_loader, self.test_loader = dataloader.get_dataloader(self.batch_size, self.test_batch_size, fold=fold)

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def stylizing(self, weight):
        out = None
        for i in range(4):
            data = weight[16 * i: 16 * i + 16]
            temp = self.stylizer.train(data)
            if i == 0:
                out = temp
            else:
                out = torch.cat((out, temp))
        return out

    def train(self):
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device).float()
            self.optimizer.zero_grad()
            prediction = self.model(data)
            target = torch.max(target, 1)[1]
            loss = self.criterion(prediction, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_correct += np.sum(torch.max(prediction, 1)[1].cpu().numpy() == target.cpu().numpy())
            total += data.size(0)
            # self.model.conv1.weight.data = self.stylizing(self.model.conv1.weight.data)

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
                target = torch.max(target, 1)[1]
                loss = self.criterion(prediction, target)
                test_loss += loss.item()
                test_correct += np.sum(torch.max(prediction, 1)[1].cpu().numpy() == target.cpu().numpy())
                total += data.size(0)
                progress_bar(batch_num, len(self.test_loader), 'test loss: %.4f | accuracy: %.4f'
                             % (test_loss / (batch_num + 1), test_correct / total))
        return test_loss / total, test_correct / total

    def save_data(self, t_a, t_l, s_a, s_l):
        result_dir = './ont_hot_pretrained' if self.pretrained else './ont_hot'
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

            # self.save_data(train_accuracy, train_loss, test_accuracy, test_loss)


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
        self.model = alexnet(pretrained=self.pretrained).to(self.device)
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

    def run(self):
        self.build_model()
        result_dir = './multi_hot_pretrained' if self.pretrained else './multi_hot'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for epoch in range(1, self.epochs + 1):
            fold = (epoch - 1) % 5
            print("\n===> Epoch {} starts: (fold: {})".format(epoch, fold))
            self.build_dataloader(fold)

            train_loss, train_accuracy = self.train()
            test_loss, test_accuracy = self.test()

            # train_acc = {'Train Accuracy': [train_accuracy]}
            # train_loss = {'Train Loss': [train_loss]}
            # test_acc = {'Test Accuracy': [test_accuracy]}
            # test_loss = {'Test Loss': [test_loss]}
            # record_info(train_acc, result_dir + '/train_acc.csv')
            # record_info(train_loss, result_dir + '/train_loss.csv')
            # record_info(test_acc, result_dir + '/test_acc.csv')
            # record_info(test_loss, result_dir + '/test_loss.csv')
