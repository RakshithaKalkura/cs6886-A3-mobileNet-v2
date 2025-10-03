import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset.data_loader import get_dataloaders
from dataset.getdataloader import GetCifar10
from models.models import MobileNetV2
from utils import AverageMeter, accuracy, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='MobileNetV2 CIFAR-10 Training')
    parser.add_argument('--data-dir', default='/content/drive/MyDrive/cifar10/', type=str)
    parser.add_argument('--out-dir', default='/content/drive/MyDrive/checkpoints_v2/', type=str)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=4e-5, type=float)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--width-mult', default=1.0, type=float)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            prec1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))

    return losses.avg, top1.avg

#Plotting loss and accurcy curves
def plot_curves(train_losses, val_losses, train_acc, val_acc, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure()
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'])
    plt.title('Loss curves')
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc)
    plt.plot(epochs, val_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.legend(['train', 'val'])
    plt.title('Accuracy curves')
    plt.savefig(os.path.join(out_dir, 'acc_curve.png'))
    plt.close()


def main():
    args = parse_args()
    class Cfg:
        data_dir = args.data_dir
        batch_size = args.batch_size
        num_workers = args.workers

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  

    # train_loader, val_loader = get_dataloaders(Cfg) # simpler data loader
    train_loader, val_loader = GetCifar10(Cfg) # better data loader with augmentations

    model = MobileNetV2(num_classes=10, width_mult=args.width_mult, dropout=args.dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        since = time.time()
        train_loss, train_top1 = train_one_epoch(train_loader, model, criterion, optimizer, device, epoch)
        val_loss, val_top1 = validate(val_loader, model, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_top1)
        val_accs.append(val_top1)

        is_best = val_top1 > best_acc
        best_acc = max(best_acc, val_top1)

        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict()},
                        is_best, args.out_dir, filename=f'checkpoint_epoch_{epoch}.pth.tar')

        print(f"Epoch: {epoch}/{args.epochs}  Train Loss: {train_loss:.4f}  Train Acc: {train_top1:.2f}%  Val Loss: {val_loss:.4f}  Val Acc: {val_top1:.2f}%  Best: {best_acc:.2f}%  Time: {time.time()-since:.1f}s")

    plot_curves(train_losses, val_losses, train_accs, val_accs, args.out_dir)

    print(f"Training finished. Best Val Acc: {best_acc:.2f}%\nOutputs saved to {args.out_dir}")


if __name__ == '__main__':
    main()

