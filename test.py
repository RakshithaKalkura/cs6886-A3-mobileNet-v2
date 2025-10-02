import argparse
import torch
import torch.nn as nn

from data_loader import get_dataloaders
from models import MobileNetV2
from dataset.getdataloader import GetCifar10
from utils import load_checkpoint, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Test MobileNetV2 CIFAR-10')
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--width-mult', default=1.0, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    class Cfg:
        data_dir = args.data_dir
        batch_size = args.batch_size
        num_workers = args.workers

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # _, test_loader = get_dataloaders(Cfg) # simpler data loader
    _, test_loader = GetCifar10(Cfg) # better data loader with augmentations

    model = MobileNetV2(num_classes=10, width_mult=args.width_mult)
    model = model.to(device)

    checkpoint = load_checkpoint(args.checkpoint, model)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_top1 = 0.0
    n = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            prec1 = accuracy(outputs, targets, topk=(1,))[0]
            total_loss += loss.item() * images.size(0)
            total_top1 += prec1.item() * images.size(0)
            n += images.size(0)

    avg_loss = total_loss / n
    avg_top1 = total_top1 / n
    print(f"Test Loss: {avg_loss:.4f}  Test Top-1 Accuracy: {avg_top1:.2f}%")


if __name__ == '__main__':
    main()

