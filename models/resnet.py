import torch.nn as nn
import torchvision.models.resnet as resnet


def get_model(pretrained=False):
    net = resnet.resnet50(pretrained=pretrained, num_classes=1000)
    net.fc = nn.Linear(2048, 10)
    return net