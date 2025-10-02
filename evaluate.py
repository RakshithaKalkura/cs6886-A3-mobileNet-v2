"""
Evaluation helper used by compression modules.
"""
import torch
from utils import get_device

def evaluate(model, dataloader, device=None):
    device = device or get_device()
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device); labels = labels.to(device)
            out = model(imgs)
            pred = out.argmax(1)
            correct += (pred==labels).sum().item(); total += labels.size(0)
    return 100.0 * correct / total