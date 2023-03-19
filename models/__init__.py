import torch
from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn

from models.resnet import ResNet
from models.shakeshake.shake_resnet import ShakeResNet
from models.wideresnet import WideResNet
from models.shakeshake.shake_resnext import ShakeResNeXt

def get_model(conf, num_class=10):
    name = conf.net_name

    if name == 'resnet50':
        model = ResNet(dataset='imagenet', depth=50, num_classes=num_class, bottleneck=True)
    elif name == 'resnet101':
        model = ResNet(dataset='imagenet', depth=101, num_classes=num_class, bottleneck=True)
    elif name == 'resnet200':
        model = ResNet(dataset='imagenet', depth=200, num_classes=num_class, bottleneck=True)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_class)

    elif name == 'shakeshake26_2x32d':
        model = ShakeResNet(26, 32, num_class)
    elif name == 'shakeshake26_2x64d':
        model = ShakeResNet(26, 64, num_class)
    elif name == 'shakeshake26_2x96d':
        model = ShakeResNet(26, 96, num_class)
    elif name == 'shakeshake26_2x112d':
        model = ShakeResNet(26, 112, num_class)
    elif name == 'shakeshake26_2x96d_next':
        model = ShakeResNeXt(26, 96, 4, num_class)
    else:
        raise NameError('no model named, %s' % name)

    return model


def num_class(dataset):
    return {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]
