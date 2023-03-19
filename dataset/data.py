import logging
import os
import torch
import torchvision
from PIL import Image

from torchvision.transforms import transforms
from augment.augmentations import Lighting, RandAugment, CutoutDefault
from dataset import imagenet_data_dali
from tools.utils import fast_collate

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def cifar_dataloader(config, augment, data_path):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),])

    if augment.randaugment.if_use:
        transform_train.transforms.insert(0, 
                            RandAugment(augment.randaugment.N, augment.randaugment.M))

    if augment.cutout > 0:
        transform_train.transforms.append(CutoutDefault(augment.cutout))

    if config.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    elif config.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    else:
        raise ValueError('invalid dataset name=%s' % config.dataset)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True , num_workers=32, pin_memory=True,
          drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=True,
        drop_last=False
    )
    return  train_loader, val_loader

def get_dataloaders(config, augment, world_size, local_rank, data_path):
    if 'cifar' in config.dataset:
        train_loader, val_loader = cifar_dataloader(config, augment, data_path)
    elif 'imagenet' in config.dataset:
        train_loader, val_loader = imagenet_data_dali.get_data_loader(
            config, world_size, local_rank, data_path
        )
    else: 
        raise ValueError('dataset=%s' % config.dataset)
    return train_loader, val_loader