"""
Sweep orchestration: to run several compression configs, evaluate accuracy and size, log to wandb.
"""
import json
import argparse
import numpy as np
import torch
from models.models import MobileNetV2
from dataset.getdataloader import GetCifar10
from deep_compress.utils import get_device, load_checkpoint
from deep_compress.size_accounting import compute_model_size
from deep_compress.centroid import CentroidRegistry, apply_centroids_to_model
from evaluate import evaluate
import deep_compress.quantize as qmod

def run_one(cfg, device, train_loader, val_loader, ckpt_path):
    model = MobileNetV2().to(device)
    ck = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ck['state_dict'])
    mask = ck.get('mask', {})
    # apply mask
    for name, p in model.named_parameters():
        if name in mask:
            p.data.mul_(torch.from_numpy(mask[name]).to(device))
    # build meta by calling compress.quantize.py logic or import that function if preferred
    # For brevity we re-use quantize.build_meta if available; else run compress/quantize.py to create meta
    bits_map = {'features': cfg.get('bits_conv',8), 'classifier': cfg.get('bits_fc',5)}
    meta = qmod.build_meta(model, mask, bits_map)
    # centroid registry
    registry = CentroidRegistry(meta, mask, device=device)
    # centroid finetune
    if cfg.get('centroid_finetune_epochs',0)>0 and train_loader is not None:
        cent_params = [getattr(registry, registry.layers[n]['cent_name']) for n in registry.layers]
        opt = torch.optim.SGD(cent_params, lr=cfg.get('centroid_lr',1e-2), momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()
        for ep in range(cfg.get('centroid_finetune_epochs')):
            for imgs,labels in train_loader:
                imgs=imgs.to(device); labels=labels.to(device)
                apply_centroids_to_model(model, registry)
                opt.zero_grad()
                out = model(imgs)
                loss = loss_fn(out, labels)
                loss.backward(); opt.step()
    apply_centroids_to_model(model, registry)
    acc = evaluate(model, val_loader, device)
    size_info = compute_model_size(meta, include_mask=True, index_diff_map={'features':8,'classifier':5})
    return {'acc': acc, 'size_bytes': size_info['total_bytes'], 'per_layer': size_info['per_layer']}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='JSON config with list of sweep configs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-dir', default='/content/drive/MyDrive/cifar10/', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--ckpt', default='/content/drive/MyDrive/checkpoints_v2/model_best.pth.tar', help='Initial full precision checkpoint to compress and evaluate')
    args=parser.parse_args()
    device = get_device(args.device)
    class cfg_data:
        data_dir = args.data_dir
        batch_size = args.batch_size
        num_workers = args.workers
    train_loader, val_loader = GetCifar10(cfg_data) #specify appropriate data_dir
    # load sweep configs
    with open(args.cfg) as f:
        cfgs = json.load(f)
    results = []
    for c in cfgs:
        res = run_one(c, device, train_loader, val_loader)
        print('cfg', c, '=>', res)
        results.append({'config':c,'result':res})
    with open('/content/drive/MyDrive/deepcompress_ckpt/sweep_out.json','w') as f:
        json.dump(results, f, indent=2)