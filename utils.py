import os
import torch
import shutil
import time
import numpy as np


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth.tar'):
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(out_dir, 'model_best.pth.tar'))


def load_checkpoint(path, model, optimizer=None):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint
    else:
        raise FileNotFoundError(f"No checkpoint found at '{path}'")
    
def get_device(pref='cuda'):
    if pref=='cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

