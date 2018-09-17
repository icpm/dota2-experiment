import torchvision.models.resnet as resnet


def get_model(pretrained=False):
    return resnet.resnet152(pretrained=pretrained, num_classes=10)
