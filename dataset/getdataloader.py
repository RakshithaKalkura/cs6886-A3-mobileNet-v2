from textwrap import fill
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
from Preprocess.augment import Cutout, CIFAR10Policy

DIR = {
    'CIFAR10': r'/content/drive/MyDrive/CIFAR10',
}
#better cifar10 dataloader with cutout and autoaugment 
def GetCifar10(batchsize, attack=False):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    if attack:
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=False)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=False)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4)
    return train_dataloader, test_dataloader


