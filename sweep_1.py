"""
sweep.py

Orchestrates pruning + quantization sweeps, centroid fine-tuning (QAT), activation profiling,
size accounting, Huffman packaging, WandB logging, and post-processing summary.

Usage:
    python sweep.py --cfg sweep_configs.json --device cuda --use_wandb

Outputs:
    - sweep_out/<run_name>_result.json   (per-run detailed metrics)
    - sweep_out/<run_name>_huff_meta.npz  (Huffman package produced)
    - sweep_out/<run_name>_huff_codes.json
    - sweep_summary.json (summary & best-runs)
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

# local project imports - ensure paths are correct
from models.models import MobileNetV2
from dataset.getdataloader import GetCifar10
from deep_compress.utils import get_device, load_checkpoint, save_checkpoint
from deep_compress.quantize import build_meta  # builds codebook+indices meta
from deep_compress.prune import make_global_mag_prune_mask, apply_mask
from deep_compress.centroid import CentroidRegistry, apply_centroids_to_model
from deep_compress.size_accounting import compute_model_size
from deep_compress.huffman import save_huffman_package, load_meta_npz
from evaluate import evaluate

# Optional WandB
try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False

# ---------------- ActivationStatsCollector (embedded) ----------------
class ActivationStatsCollector:
    """
    Simple forward-hook-based activation collector to estimate average FP32 activation
    bytes per sample per layer over a small validation subset.
    """
    def __init__(self, model, modules_to_monitor=None):
        self.model = model
        self.modules_to_monitor = set(modules_to_monitor) if modules_to_monitor else None
        self.handles = []
        self.accum = {}  # name -> {'sum_bytes':float, 'count':int}

    def _make_hook(self, name):
        def hook(module, inp, out):
            # out may be tensor or tuple/list of tensors
            total_bytes = 0
            total_count = 0
            if isinstance(out, torch.Tensor):
                t = out.detach().cpu()
                total_bytes += t.numel() * 4
                total_count += t.shape[0]
            elif isinstance(out, (list, tuple)):
                for tt in out:
                    if isinstance(tt, torch.Tensor):
                        tt = tt.detach().cpu()
                        total_bytes += tt.numel() * 4
                        total_count += tt.shape[0]
            if name not in self.accum:
                self.accum[name] = {'sum_bytes': 0.0, 'count': 0}
            self.accum[name]['sum_bytes'] += float(total_bytes)
            self.accum[name]['count'] += int(total_count)
        return hook

    def add_hooks(self):
        for name, module in self.model.named_modules():
            if (self.modules_to_monitor is None) or (name in self.modules_to_monitor):
                h = module.register_forward_hook(self._make_hook(name))
                self.handles.append(h)

    def remove_hooks(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def reset(self):
        self.accum = {}

    def summary(self):
        # returns dict name->avg_fp32_bytes_per_sample
        out = {}
        for name, v in self.accum.items():
            if v['count'] == 0:
                out[name] = 0.0
            else:
                out[name] = v['sum_bytes'] / v['count']
        return out

# ---------------- small helpers ----------------
def count_parameters_bytes(model):
    return sum(p.numel() for p in model.parameters()) * 4

def bytes_for_activations_summary(act_summary_fp32, act_bits, per_layer_scale_bytes=4):
    total_fp32 = sum(act_summary_fp32.values())
    total_quant = 0
    per_layer_quant = {}
    for lname, bytes_fp32 in act_summary_fp32.items():
        floats = bytes_fp32 / 4.0
        q_bytes = int(np.ceil((floats * act_bits) / 8.0))
        q_bytes += per_layer_scale_bytes
        per_layer_quant[lname] = {'fp32_bytes': bytes_fp32, 'quant_bytes': q_bytes}
        total_quant += q_bytes
    return total_fp32, total_quant, per_layer_quant

# ---------------- single-run pipeline ----------------
def run_one(cfg, train_loader, val_loader, device, out_dir='sweep_out'):
    start_time = time.time()
    os.makedirs(out_dir, exist_ok=True)
    name = cfg.get('name', f"run_{int(start_time)}")
    print(f"\n=== Starting run {name} ===")

    # 1) load baseline model checkpoint
    model = MobileNetV2().to(device)
    ck = load_checkpoint(cfg['baseline_ckpt'], map_location=device)
    # ck may contain {'state_dict':..., ...} or be a bare state_dict
    if isinstance(ck, dict) and 'state_dict' in ck:
        model.load_state_dict(ck['state_dict'])
    else:
        try:
            model.load_state_dict(ck)
        except Exception:
            # last resort: try key 'state'
            model.load_state_dict(ck.get('state', ck))

    # baseline accuracy
    acc_baseline = evaluate(model, val_loader, device)
    print("Baseline accuracy (no compression):", acc_baseline)

    # 2) pruning
    sparsity = float(cfg.get('sparsity', 0.8))
    mask, threshold = make_global_mag_prune_mask(model, sparsity)
    apply_mask(model, mask)
    # optional prune finetune
    if int(cfg.get('prune_finetune_epochs', 0)) > 0:
        print("Running prune finetune for", cfg['prune_finetune_epochs'], "epochs")
        opt = torch.optim.SGD(model.parameters(), lr=cfg.get('prune_lr', 1e-2), momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        for ep in range(int(cfg.get('prune_finetune_epochs'))):
            model.train()
            for imgs, labels in train_loader:
                imgs = imgs.to(device); labels = labels.to(device)
                opt.zero_grad()
                out = model(imgs)
                loss = loss_fn(out, labels)
                loss.backward()
                # zero grads where masked
                for n,p in model.named_parameters():
                    if n in mask and p.grad is not None:
                        p.grad.mul_(mask[n].to(p.device))
                opt.step()
                # reapply mask
                for n,p in model.named_parameters():
                    if n in mask:
                        p.data.mul_(mask[n].to(p.device))
    # 3) build quant meta
    bits_map = {'features': int(cfg.get('bits_conv', 8)), 'classifier': int(cfg.get('bits_fc', 5))}
    meta = build_meta(model, mask, bits_map)  # dict layer -> {'codebook','indices','shape','bitwidth','nonzeros'}

    # 4) centroid fine-tune (QAT)
    reg = CentroidRegistry(meta, mask, device=device)
    apply_centroids_to_model(model, reg)  # set model weights to quantized reconstruction
    if int(cfg.get('centroid_finetune_epochs', 0)) > 0:
        print("Running centroid fine-tune for", cfg['centroid_finetune_epochs'], "epochs")
        cent_params = []
        for n in reg.layers:
            cent_params.append(getattr(reg, reg.layers[n]['cent_name']))
        opt = torch.optim.SGD(cent_params, lr=cfg.get('centroid_lr', 1e-2), momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        for ep in range(int(cfg.get('centroid_finetune_epochs'))):
            model.train()
            for imgs, labels in train_loader:
                imgs = imgs.to(device); labels = labels.to(device)
                apply_centroids_to_model(model, reg)  # reconstruct before forward
                opt.zero_grad()
                out = model(imgs)
                loss = loss_fn(out, labels)
                loss.backward()
                # centroids will have grads because reconstruct used centroids
                opt.step()

    apply_centroids_to_model(model, reg)
    acc_compressed = evaluate(model, val_loader, device)
    print("Compressed model accuracy:", acc_compressed)

    # 5) Activation profiling (small validation subset)
    act_col = ActivationStatsCollector(model)
    act_col.add_hooks()
    model.eval()
    with torch.no_grad():
        n = 0
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            _ = model(imgs)
            n += 1
            if n >= int(cfg.get('act_batches', 5)):
                break
    act_col.remove_hooks()
    act_summary_fp32 = act_col.summary()
    act_bits = int(cfg.get('act_bits', 8))
    act_fp32_bytes_per_sample, act_quant_bytes_per_sample, per_layer_act_quant = bytes_for_activations_summary(act_summary_fp32, act_bits)

    # 6) Size accounting
    size_info = compute_model_size(meta, include_mask=True, index_diff_map={'features': 8, 'classifier': 5})
    compressed_bytes_raw = int(size_info['total_bytes'])
    original_bytes = int(count_parameters_bytes(model))
    weights_only_bytes = compressed_bytes_raw
    ratio_model_raw = compressed_bytes_raw / original_bytes
    ratio_weights_raw = weights_only_bytes / original_bytes
    ratio_activations = (act_quant_bytes_per_sample / act_fp32_bytes_per_sample) if act_fp32_bytes_per_sample > 0 else float('nan')
    final_mb_raw = compressed_bytes_raw / 1e6

    # 7) Huffman packaging (empirical)
    h_npz, h_codes = save_huffman_package(meta, out_prefix=os.path.join(out_dir, name + '_huff'))
    # load to compute empirical bits
    h_archive = np.load(h_npz, allow_pickle=True)
    empirical_bits = 0
    for k in h_archive.files:
        if k.endswith('__bits'):
            empirical_bits += int(h_archive[k])
    # compute codebook bytes (float32)
    codebook_bytes_total = 0
    for k in h_archive.files:
        if k.endswith('__cb'):
            codebook_bytes_total += int(h_archive[k].nbytes)
    compressed_bytes_huffman = (empirical_bits + 7)//8 + codebook_bytes_total
    final_mb_huffman = compressed_bytes_huffman / 1e6
    ratio_model_huffman = compressed_bytes_huffman / original_bytes

    # 8) prepare result dict
    result = {
        'name': name,
        'config': cfg,
        'acc_baseline': float(acc_baseline),
        'acc_compressed': float(acc_compressed),
        'original_bytes': int(original_bytes),
        'compressed_bytes_raw': int(compressed_bytes_raw),
        'compressed_bytes_huffman_est': int(compressed_bytes_huffman),
        'final_MB_raw': float(final_mb_raw),
        'final_MB_huffman': float(final_mb_huffman),
        'ratio_model_raw': float(ratio_model_raw),
        'ratio_model_huffman': float(ratio_model_huffman),
        'ratio_weights_raw': float(ratio_weights_raw),
        'ratio_activations': float(ratio_activations),
        'act_fp32_bytes_per_sample': float(act_fp32_bytes_per_sample),
        'act_quant_bytes_per_sample': float(act_quant_bytes_per_sample),
        'per_layer_act_quant': per_layer_act_quant,
        'size_breakdown': size_info['per_layer'],
        'empirical_huffman_bits': int(empirical_bits),
        'huffman_meta_path': h_npz,
        'huffman_codes_path': h_codes,
        'elapsed_sec': time.time() - start_time
    }

    # save per-run result JSON
    out_path = os.path.join(out_dir, f"{name}_result.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print("Saved run result to", out_path)

    # WandB logging (optional)
    if cfg.get('use_wandb', True) and _WANDB:
        run = wandb.init(project=cfg.get('wandb_project', 'deep_compress'), config=cfg, reinit=True)
        print("WandB run created:", run.id)
        print("WandB run URL:", run.get_url()) 
        # log scalar metrics
        wandb.log({
            'acc_baseline': result['acc_baseline'],
            'acc_compressed': result['acc_compressed'],
            'final_MB_huffman': result['final_MB_huffman'],
            'ratio_model_huffman': result['ratio_model_huffman'],
            'ratio_weights_raw': result['ratio_weights_raw'],
            'ratio_activations': result['ratio_activations'],
            'empirical_huffman_bits': result['empirical_huffman_bits']
        })
        # log per-layer breakdown as table (optional)
        try:
            import pandas as pd
            rows = []
            for lname, info in result['size_breakdown'].items():
                row = {'layer': lname}
                row.update(info)
                rows.append(row)
            tbl = pd.DataFrame(rows)
            wandb.log({"size_breakdown_table": wandb.Table(dataframe=tbl)})
        except Exception:
            pass
        wandb.finish()

    return result

# ---------------- run sweep and postprocess ----------------
def run_sweep(cfg_path, device, train_loader=None, val_loader=None, out_dir='sweep_out'):
    with open(cfg_path) as f:
        cfgs = json.load(f)
    results = []
    for cfg in cfgs:
        r = run_one(cfg, train_loader, val_loader, device, out_dir=out_dir)
        results.append(r)
    # save combined
    combined_path = os.path.join(out_dir, 'sweep_combined.json')
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved combined sweep results to", combined_path)

    # postprocessing: find best ratios subject to tolerance on accuracy
    tol = float(0.0)  # default: exact or change to e.g. 1.0 (percent) to allow small drop
    # we will derive baseline accuracy per run and allow drop tol
    best_model = None
    best_weights = None
    best_acts = None
    for r in results:
        base = r['acc_baseline']
        acc = r['acc_compressed']
        # require compressed acc >= base - tol
        if acc >= base - tol:
            # model (use huffman metric)
            mr = r['ratio_model_huffman']
            if (best_model is None) or (mr < best_model['ratio']):
                best_model = {'ratio': mr, 'run': r}
            # weights
            wr = r['ratio_weights_raw']
            if (best_weights is None) or (wr < best_weights['ratio']):
                best_weights = {'ratio': wr, 'run': r}
            # activations
            ar = r['ratio_activations']
            if (best_acts is None) or (ar < best_acts['ratio']):
                best_acts = {'ratio': ar, 'run': r}

    summary = {
        'runs': results,
        'best_model_under_tol': best_model,
        'best_weights_under_tol': best_weights,
        'best_activations_under_tol': best_acts
    }
    with open('sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved sweep_summary.json")
    return summary

# ---------------- CLI ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='Path to sweep_configs.json (list of configs)')
    parser.add_argument('--device', default='cuda', help='Device to run on')
    parser.add_argument('--use_wandb', action='store_true', help='Log runs to WandB (must be logged in)')
    parser.add_argument('--out', default='/content/drive/MyDrive/deepcompress_ckpt/sweep_out_v1', help='Output directory for results')
    parser.add_argument('--data-dir', default='/content/drive/MyDrive/cifar10/', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    args = parser.parse_args()

    device = get_device(args.device)
    class cfg_data:
        data_dir = args.data_dir
        batch_size = args.batch_size
        num_workers = args.workers
    train_loader, test_loader = GetCifar10(cfg_data) #specify appropriate data_dir

    # load sweep configs and run
    summary = run_sweep(args.cfg, device, train_loader=train_loader, val_loader=test_loader, out_dir=args.out)
    print("Sweep finished. Summary keys:", summary.keys())
