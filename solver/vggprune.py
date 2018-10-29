import os

from torchvision import datasets, transforms

from models import *


class VGGPrune(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.model = None
        self.new_model = None
        self.pruned_ratio = None
        self.cfg = None
        self.cfg_mask = None

    def initialize(self):
        if not os.path.exists(self.args.save):
            os.makedirs(self.args.save)

        self.model = vgg(dataset=self.args.dataset, depth=self.args.depth).to(self.device)
        if self.args.model:
            if os.path.isfile(self.args.model):
                print("=> loading checkpoint '{}'".format(self.args.model))
                checkpoint = torch.load(self.args.model)
                self.args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                      .format(self.args.model, checkpoint['epoch'], best_prec1))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

    def pre_process(self):
        total = 0
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index + size)] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * self.args.percent)
        thre = y[thre_index]

        pruned = 0
        self.cfg = []
        self.cfg_mask = []
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre.to(self.device)).float().cuda()
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                self.cfg.append(int(torch.sum(mask)))
                self.cfg_mask.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                      format(k, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(m, nn.MaxPool2d):
                self.cfg.append('M')

        self.pruned_ratio = pruned / total

        print('Pre-processing Successful!')

    def prune(self):
        self.new_model = vgg(dataset=self.args.dataset, cfg=self.cfg).to(self.device)

        num_parameters = sum([param.nelement() for param in self.new_model.parameters()])
        savepath = os.path.join(self.args.save, "prune.txt")
        acc = self.test(self.model)

        with open(savepath, "w") as fp:
            fp.write("Configuration: \n" + str(self.cfg) + "\n")
            fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
            fp.write("Test accuracy: \n" + str(acc))

        layer_id_in_cfg = 0
        start_mask = torch.ones(3)
        end_mask = self.cfg_mask[layer_id_in_cfg]
        for [m0, m1] in zip(self.model.modules(), self.new_model.modules()):
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(self.cfg_mask):  # do not change in Final FC
                    end_mask = self.cfg_mask[layer_id_in_cfg]
            elif isinstance(m0, nn.Conv2d):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
            elif isinstance(m0, nn.Linear):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0].clone()
                m1.bias.data = m0.bias.data.clone()

        torch.save({'cfg': self.cfg, 'state_dict': self.new_model.state_dict()}, os.path.join(self.args.save, 'pruned.pth.tar'))

        print(self.new_model)
        self.model = self.new_model
        self.test(self.model)

    def test(self, model):
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else {}
        if self.args.dataset == 'cifar10':
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                batch_size=self.args.test_batch_size, shuffle=True, **kwargs)
        elif self.args.dataset == 'cifar100':
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                batch_size=self.args.test_batch_size, shuffle=True, **kwargs)
        else:
            raise ValueError("No valid dataset is given.")
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        return correct / float(len(test_loader.dataset))

    def run(self):
        self.initialize()
        self.pre_process()
        self.prune()