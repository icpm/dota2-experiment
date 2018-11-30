import torch
import torch.nn as nn


def prune(_model, percent=0.6):
    total = 0
    for m in _model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()

    conv_weights = torch.zeros(total).cuda()
    index = 0
    for m in _model.modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.numel()
            conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    thre_index = int(total * percent)
    thre = y[thre_index]

    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    for k, m in enumerate(_model.modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned += mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                  format(k, mask.numel(), int(torch.sum(mask))))
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned / total))

    return _model, pruned, total


def get_parameter_num(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()
    return total