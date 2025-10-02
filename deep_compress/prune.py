"""
Magnitude pruning module. Outputs mask dict and pruned checkpoint.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from utils import load_checkpoint, save_checkpoint, get_device
from models.models import MobileNetV2

# basic global magnitude pruning - create masks and apply

def make_global_mag_prune_mask(model, sparsity):
    # collect weights
    vals = torch.cat([p.detach().abs().view(-1) for n,p in model.named_parameters() if 'weight' in n])
    k_keep = int(vals.numel() * (1.0 - sparsity))
    if k_keep < 1:
        threshold = float('inf')
    else:
        threshold = vals.kthvalue(k_keep).values.item()
    mask = {}
    for name, p in model.named_parameters():
        if 'weight' in name:
            mask[name] = (p.detach().abs() >= threshold).float()
    return mask, threshold

def apply_mask(model, mask):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in mask:
                p.mul_(mask[name].to(p.device))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--sparsity', type=float, default=0.8)
    parser.add_argument('--out', default='/content/drive/MyDrive/deepcompress_ckpt/pruned_ckpt.pth')
    parser.add_argument('--device', default='cuda')
    args=parser.parse_args()
    device = get_device(args.device)
    model = MobileNetV2().to(device)
    ck = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(ck['state_dict'])
    mask, thr = make_global_mag_prune_mask(model, args.sparsity)
    apply_mask(model, mask)
    # save mask as numpy arrays
    mask_cpu = {k: v.cpu().numpy() for k,v in mask.items()}
    save_checkpoint({'state_dict': model.state_dict(), 'mask': mask_cpu}, args.out)
    print(f"Saved pruned model to {args.out} (threshold {thr:.6e})")

if __name__=='__main__':
    main()