import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#a simple data loader for CIFAR-10 
def build_train_transforms():
    # CIFAR-10 mean/std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform


def build_test_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform


def get_dataloaders(cfg):
    train_transform = build_train_transforms()
    test_transform = build_test_transforms()

    train_dataset = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, test_loader

