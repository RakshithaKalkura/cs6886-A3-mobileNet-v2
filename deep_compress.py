"""
deep_compress_pipeline.py

The code is implemented based on the Deep Compression paper: "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding"

Deep compression pipeline for PyTorch:
- Pruning (magnitude) with mask-based training
- Weight quantization: k-means weight-sharing + centroid fine-tuning (QAT)
- Activation quantization (fake-quant / per-tensor or per-channel)
- Huffman code helper to optionally compress indices
- Full size accounting (weights, indices, indexes metadata, codebook, activation scales)
- Sweep runner + wandb logging, including Parallel Coordinates

Requirements:
    torch, torchvision, sklearn, numpy, wandb (optional for logging)
Usage:
  1) Train baseline
  2) Run prune stage: python deep_compress.py --stage prune --ckpt baseline.pth --sparsity 0.8
  3) Run quantize stage (k-means + centroid finetune): python deep_compress.py --stage quantize --ckpt pruned.pth --bits_conv 8 --bits_fc 5
  4) Run activation quant stage: python deep_compress.py --stage act-quant --ckpt quantized.pth --act-bits 8
  5) Run huffman stage to produce codebook+encoded stream: python deep_compress.py --stage huffman --ckpt act_quant.pth --out-dir results
  6) Run sweep: python deep_compress.py --stage sweep ...
"""

import math, json, os, argparse, copy, sys
from collections import Counter, defaultdict
import heapq
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from data_loader import get_dataloaders
from dataset.getdataloader import GetCifar10
from models import MobileNetV2
from utils import AverageMeter, accuracy, save_checkpoint

# If wandb not available, we fallback
try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False

# ------------ Utilities ------------
def count_parameters_bytes(model):
    """Return bytes needed to store full-precision parameters (float32)"""
    total = 0
    for p in model.parameters():
        total += p.numel() * 4  # float32 -> 4 bytes
    return total

def bits_to_bytes(b):
    return (b + 7) // 8

# ------------ Pruning (magnitude) ------------
def make_global_mag_prune_mask(model, sparsity):
    """
    Compute mask dictionary mapping param_name -> mask tensor (1 keep, 0 prune)
    sparsity in (0,1) fraction pruned (e.g., 0.8 => 80% weights set to zero)
    Only applied to weight tensors (not biases)
    """
    # collect absolute values of all weights
    all_vals = []
    name_and_shape = []
    for name, p in model.named_parameters():
        if 'weight' in name and p.requires_grad:
            arr = p.detach().cpu().abs().view(-1)
            all_vals.append(arr)
            name_and_shape.append((name, p.shape))
    if len(all_vals) == 0:
        return {}
    all_concat = torch.cat(all_vals)
    k_keep = int(all_concat.numel() * (1.0 - sparsity))
    if k_keep < 1:
        threshold = float('inf')
    else:
        threshold = all_concat.kthvalue(k_keep).values.item()
    mask = {}
    for name, p in model.named_parameters():
        if 'weight' in name and p.requires_grad:
            m = (p.detach().abs() >= threshold).float()
            mask[name] = m.to(p.device)
    return mask, threshold

def apply_mask_to_model(model, mask):
    """Zero out pruned weights in model in-place"""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in mask:
                p.mul_(mask[name])

# A helper training step that respects masks (zero grads where mask==0 and reapplies mask)
def train_one_epoch_with_mask(model, loader, optimizer, criterion, mask, device):
    model.train()
    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        # zero grads for pruned params
        if mask:
            for name, p in model.named_parameters():
                if name in mask and p.grad is not None:
                    p.grad.mul_(mask[name])
        optimizer.step()
        # reapply mask to keep zeros exactly zero
        if mask:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in mask:
                        p.mul_(mask[name])

# ------------ Weight quantization (k-means weight sharing) ------------
def kmeans_cluster_1d(values, n_clusters, random_state=0, n_init=10):
    """values: 1d numpy array, returns centroids (k,) and labels (len(values),)"""
    if values.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    k = min(n_clusters, len(values))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(values.reshape(-1,1))
    centroids = kmeans.cluster_centers_.squeeze().astype(np.float32)
    labels = kmeans.labels_.astype(np.int32)
    return centroids, labels

def build_weight_sharing_metadata(model, mask_dict, bits_map):
    """
    For each weight tensor (name), produce:
       - codebook (float32 numpy array)
       - indices: same shape as param but with -1 for pruned entries, else int index into codebook
       - bitwidth (b)
    bits_map: list of (substr, bits) e.g. [('classifier',5),('features',8)]
    """
    meta = {}
    device = next(model.parameters()).device
    for name, p in model.named_parameters():
        if 'weight' not in name or not p.requires_grad:
            continue
        mask = mask_dict.get(name, torch.ones_like(p)).cpu().numpy()
        w = p.detach().cpu().numpy()
        flat = w.flatten()
        nz_idx = np.where(mask.flatten() != 0)[0]
        if nz_idx.size == 0:
            meta[name] = {'codebook': np.array([], dtype=np.float32),
                          'indices': -1 * np.ones_like(flat, dtype=np.int32),
                          'shape': p.shape, 'bitwidth': 1, 'nonzeros': 0}
            continue
        vals = flat[nz_idx]
        # choose bitwidth by matching substrings
        b = None
        for substr, bits in bits_map.items():
            if substr in name:
                b = bits
                break
        if b is None:
            b = 8
        k = max(2, 2 ** b)
        centroids, labels = kmeans_cluster_1d(vals, k)
        indices_full = -1 * np.ones_like(flat, dtype=np.int32)
        indices_full[nz_idx] = labels
        meta[name] = {'codebook': centroids, 'indices': indices_full.reshape(p.shape),
                       'shape': p.shape, 'bitwidth': b, 'nonzeros': int(nz_idx.size)}
    return meta

# ------------ Quantized wrappers & centroid fine-tuning (QAT) ------------
class CentroidRegistry(nn.Module):
    """
    Holds centroid tensors for all layers as trainable nn.Parameters.
    We reconstruct param tensors each forward/step by replacing non-pruned entries using indices.
    """
    def __init__(self, meta, mask_dict, device='cpu'):
        super().__init__()
        self.device = device
        self.layers = {}
        # For each layer with centroids: register Parameter(centroids) and buffers: indices, mask, shape
        for name, info in meta.items():
            cb = info['codebook']
            indices = info['indices']
            shape = tuple(info['shape'])
            mask = mask_dict.get(name, torch.ones(shape)).cpu().numpy()
            if cb.size > 0:
                cent = nn.Parameter(torch.tensor(cb, dtype=torch.float32, device=device))
            else:
                cent = nn.Parameter(torch.zeros(0, dtype=torch.float32, device=device))
            # indices buffer stored as int32 numpy (on CPU) but also register as buffer if needed
            buf_idx = torch.from_numpy(indices.astype(np.int32)).to(device)
            buf_mask = torch.from_numpy(mask.astype(np.float32)).to(device)
            self.register_parameter(name.replace('.','_') + '_cent', cent)  # parameter name unique
            self.register_buffer(name.replace('.','_') + '_indices', buf_idx)
            self.register_buffer(name.replace('.','_') + '_mask', buf_mask)
            self.layers[name] = {'cent_param_name': name.replace('.','_') + '_cent',
                                 'indices_name': name.replace('.','_') + '_indices',
                                 'mask_name': name.replace('.','_') + '_mask',
                                 'shape': shape}
    def reconstruct_param(self, name):
        info = self.layers[name]
        cent = getattr(self, info['cent_param_name'])
        indices = getattr(self, info['indices_name'])
        mask = getattr(self, info['mask_name'])
        if cent.numel() == 0:
            return torch.zeros(info['shape'], device=self.device)
        idx_flat = indices.view(-1)
        # replace -1 with 0 safe index but mask will zero them
        safe_idx = idx_flat.clone()
        safe_idx[safe_idx < 0] = 0
        w_flat = cent[safe_idx].view(info['shape'])
        w_flat = w_flat * mask
        return w_flat

def apply_centroids_to_model(model, centroid_registry):
    """Overwrite model weight tensors with reconstructed quantized weights from centroids"""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'weight' in name and name in centroid_registry.layers:
                new_w = centroid_registry.reconstruct_param(name).to(p.device)
                p.data.copy_(new_w)

# ------------ Activation quantization (fake quant) ------------
class FakeQuantizeAffine(nn.Module):
    """
    Per-tensor or per-channel fake quantization module (affine): y = clamp(round(x/scale)+zp) * scale
    Stores scale (learnable or computed) and zero_point (int)
    """
    def __init__(self, bitwidth=8, per_channel=False, ch_axis=1, learn_scale=False, signed=False):
        super().__init__()
        self.bitwidth = bitwidth
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.learn_scale = learn_scale
        self.signed = signed
        # scale parameter(s)
        self.scale = nn.Parameter(torch.tensor(1.0)) if learn_scale else None
        # if per_channel and learn_scale, use Parameter of shape (C,)
        self.register_buffer('scale_buffer', torch.tensor(1.0))
    def forward(self, x):
        # compute scale if not learned: use max absolute / (2^{b-1}-1) heuristic for signed or (2^b -1) for unsigned
        if self.learn_scale:
            scale = torch.clamp(self.scale, min=1e-8)
        else:
            if self.per_channel:
                # compute per-channel max over spatial dims
                dims = list(range(x.dim()))
                dims.pop(self.ch_axis)
                max_abs = x.abs().amax(dim=dims, keepdim=True)
                scale = max_abs / (2 ** (self.bitwidth - (1 if self.signed else 0)) - 1 + 1e-12)
                scale = torch.clamp(scale, min=1e-8)
            else:
                max_abs = x.abs().max()
                scale = max_abs / (2 ** (self.bitwidth - (1 if self.signed else 0)) - 1 + 1e-12)
                scale = torch.clamp(scale, min=1e-8)
        # fake quant
        qmin = - (2**(self.bitwidth-1)) if self.signed else 0
        qmax = (2**(self.bitwidth-1)-1) if self.signed else (2**self.bitwidth - 1)
        y = x / scale
        yq = y.round().clamp(qmin, qmax)
        yfq = yq * scale
        return yfq

# ------------ Huffman coding helpers ------------
class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq; self.symbol = symbol; self.left = left; self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_map(symbols):
    freq = Counter(symbols)
    heap = [HuffmanNode(f, s) for s,f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        return {heap[0].symbol: '0'}
    while len(heap) > 1:
        a = heapq.heappop(heap); b = heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(a.freq + b.freq, None, a, b))
    root = heapq.heappop(heap)
    codes = {}
    def walk(node, prefix=''):
        if node.symbol is not None:
            codes[node.symbol] = prefix
        else:
            walk(node.left, prefix + '0'); walk(node.right, prefix + '1')
    walk(root, '')
    return codes

# ------------ Size & metadata accounting ------------
def bytes_for_codebook(codebook):
    """codebook is numpy float32 array"""
    return codebook.size * 4

def bytes_for_indices(indices, bitwidth):
    """indices: numpy array of ints (for non-pruned entries). This counts raw bit usage (no Huffman)"""
    nonneg = indices.flatten()
    # count only entries >=0 (non-pruned); pruned entries are not stored as indices (sparse)
    valid = nonneg[nonneg >= 0]
    bits_per_index = bitwidth
    total_bits = valid.size * bits_per_index
    return (total_bits + 7) // 8

def bytes_for_sparse_index_metadata(indices, index_diff_bits=8):
    """
    We will store sparse index positions as relative diffs between nonzero positions.
    index_diff_bits: how many bits to store each diff (paper uses 8 bits for conv, 5 bits for fc)
    If diff exceeds max representable, a filler zero is added (matches paper idea) — we conservatively assume
    worst-case representation: number_of_nonzeros * index_diff_bits (upper bound).
    """
    flat = indices.flatten()
    nz_positions = np.where(flat >= 0)[0]
    if nz_positions.size == 0:
        return 0
    return (nz_positions.size * index_diff_bits + 7) // 8

def compute_model_size_bytes_from_meta(meta, include_mask_bitmap=False, index_diff_bits_map=None):
    """
    meta: per-layer meta returned by build_weight_sharing_metadata
    index_diff_bits_map: dict substring->bits for relative index diffs (e.g., conv->8, fc->5)
    include_mask_bitmap: if True, include storage for mask bitmap (1 bit per param)
    Returns dict with totals in bytes and breakdown
    """
    total = 0
    breakdown = {}
    for name, m in meta.items():
        codebook = m['codebook']
        indices = m['indices']
        bitwidth = m['bitwidth']
        cb_bytes = bytes_for_codebook(codebook)
        idx_bytes = bytes_for_indices(indices, bitwidth)
        # heuristic index diff bits:
        idx_diff_bits = 8
        if index_diff_bits_map:
            for s,bits in index_diff_bits_map.items():
                if s in name:
                    idx_diff_bits = bits; break
        meta_bytes = bytes_for_sparse_index_metadata(indices, idx_diff_bits)
        mask_bytes = 0
        if include_mask_bitmap:
            mask_bits = indices.size
            mask_bytes = (mask_bits + 7) // 8
        layer_total = cb_bytes + idx_bytes + meta_bytes + mask_bytes
        breakdown[name] = {'codebook_bytes': cb_bytes, 'indices_bytes': idx_bytes,
                           'index_meta_bytes': meta_bytes, 'mask_bytes': mask_bytes,
                           'layer_total_bytes': layer_total}
        total += layer_total
    return {'total_bytes': total, 'per_layer': breakdown}

# ------------ Activation measurement (hooks) ------------
class ActivationStatsCollector:
    """
    Install forward hooks to collect activation tensors for selected modules.
    We'll collect the average size (bytes) per input sample for original float32 activation tensors,
    and compute quantized bytes if using activation_bitwidth.
    """
    def __init__(self, model, modules_to_monitor=None):
        self.model = model
        self.modules = modules_to_monitor or []
        self.handles = []
        self.accum = defaultdict(lambda: {'sum_bytes': 0.0, 'count': 0})
    def _hook(self, name):
        def fn(module, inp, out):
            # out is tensor or tuple
            if isinstance(out, torch.Tensor):
                t = out.detach().cpu()
                bytes_f32 = t.numel() * 4
                self.accum[name]['sum_bytes'] += bytes_f32
                self.accum[name]['count'] += t.shape[0]  # number of samples in batch
            elif isinstance(out, (list,tuple)):
                # sum over tensors
                s=0; c=0
                for tt in out:
                    if isinstance(tt, torch.Tensor):
                        s += tt.detach().cpu().numel()*4
                        c += tt.shape[0]
                self.accum[name]['sum_bytes'] += s
                self.accum[name]['count'] += c
        return fn
    def add_hooks(self):
        for name, module in self.model.named_modules():
            if (not self.modules) or (name in self.modules):
                h = module.register_forward_hook(self._hook(name))
                self.handles.append(h)
    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []
    def reset(self):
        self.accum = defaultdict(lambda: {'sum_bytes': 0.0, 'count': 0})
    def summary(self):
        # return per-module avg bytes per sample
        out = {}
        for name, v in self.accum.items():
            if v['count'] == 0:
                out[name] = 0.0
            else:
                out[name] = v['sum_bytes'] / v['count']
        return out

# ------------ Sweeper & WandB logging ------------
def run_sweep_and_log(model_fn, dataloaders, device, sweep_configs, out_dir='results', use_wandb=False):
    """
    sweep_configs: list of dicts each containing settings e.g. {'sparsity':0.8, 'bits_conv':8, 'bits_fc':5, 'act_bits':8}
    model_fn: function returning model instance (fresh) given num_classes
    dataloaders: (train_loader, val_loader)
    For each config:
       - load baseline model or train from scratch externally and pass ckpt path in config, or call model_fn and let user train
       - apply prune -> quantize -> centroid finetune with provided hyperparams
       - evaluate validation accuracy
       - compute sizes and per-layer stats
       - log to wandb the config and metrics
    Returns list of results dict
    """
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, cfg in enumerate(sweep_configs):
        print(f"Running sweep config {i+1}/{len(sweep_configs)}: {cfg}")
        model = model_fn().to(device)
        # if cfg has 'ckpt' load it; else assume baseline already is present or user uses model_fn preloaded
        if cfg.get('ckpt'):
            ck = torch.load(cfg['ckpt'], map_location=device)
            model.load_state_dict(ck)
        # stage 1: prune
        mask, thr = make_global_mag_prune_mask(model, cfg.get('sparsity', 0.8))
        apply_mask_to_model(model, mask)
        # fine-tune pruned model lightly 
        if cfg.get('prune_finetune_epochs',0) > 0:
            optim = torch.optim.SGD(model.parameters(), lr=cfg.get('prune_lr',1e-2), momentum=0.9, weight_decay=1e-5)
            for e in range(cfg.get('prune_finetune_epochs')):
                train_one_epoch_with_mask(model, dataloaders[0], optim, nn.CrossEntropyLoss(), mask, device)
        # stage 2: build weight-sharing meta
        bits_map = cfg.get('bits_map', {'classifier': cfg.get('bits_fc',5), 'features': cfg.get('bits_conv',8)})
        meta = build_weight_sharing_metadata(model, mask, bits_map)
        # centroid registry and QAT centroid fine-tuning (train centroids)
        registry = CentroidRegistry(meta, mask, device=device)
        # apply centroids to model initial
        apply_centroids_to_model(model, registry)
        # centroid-only optimizer
        centroid_params = [getattr(registry, registry.layers[name]['cent_param_name']) for name in registry.layers]
        optim_cent = torch.optim.SGD(centroid_params, lr=cfg.get('centroid_lr',1e-2), momentum=0.9)
        # centroid fine-tune
        for e in range(cfg.get('centroid_finetune_epochs', 10)):
            model.train()
            for imgs, labels in dataloaders[0]:
                imgs = imgs.to(device); labels = labels.to(device)
                optim_cent.zero_grad()
                # before forward ensure model weights replaced
                apply_centroids_to_model(model, registry)
                out = model(imgs)
                loss = nn.CrossEntropyLoss()(out, labels)
                loss.backward()
                # gradients on centroids will exist because reconstruct used centroids
                optim_cent.step()
        # eval
        apply_centroids_to_model(model, registry)
        val_acc = evaluate_model(model, dataloaders[1], device)
        # activation stats
        act_col = ActivationStatsCollector(model)
        act_col.add_hooks()
        # run a subset of validation to gather activation sizes
        with torch.no_grad():
            n=0
            for imgs, labels in dataloaders[1]:
                imgs = imgs.to(device)
                _ = model(imgs)
                n+=1
                if n >= cfg.get('act_batches', 10):
                    break
        act_col.remove_hooks()
        act_summary = act_col.summary()  # avg bytes per sample for each monitored module
        # compute storage bytes
        index_diff_map = cfg.get('index_diff_bits_map', {'conv':8, 'fc':5})
        size_info = compute_model_size_bytes_from_meta(meta, include_mask_bitmap=cfg.get('include_mask', True), index_diff_bits_map=index_diff_map)
        # activation compressed bytes: assume act_bits quantized uniformly
        act_bit = cfg.get('act_bits', 8)
        act_bytes_uncompressed = sum(act_summary.values())
        act_bytes_quant = 0
        for name, b in act_summary.items():
            # bytes per sample after quant = (#floats * act_bit)/8 + scale overhead (we add 4 bytes per layer)
            approx_floats = b / 4.0
            quant_bytes = (approx_floats * act_bit + 7) // 8
            act_bytes_quant += quant_bytes + 4  # 4 bytes per-layer scale overhead approx
        result = {
            'config': cfg, 'val_acc': val_acc, 'model_size_bytes': size_info['total_bytes'],
            'act_bytes_uncompressed': act_bytes_uncompressed, 'act_bytes_quant': act_bytes_quant,
            'per_layer': size_info['per_layer']
        }
        # log to wandb
        if use_wandb and _WANDB:
            wandb.init(project=cfg.get('wandb_project','deep_compress'), config=cfg, reinit=True)
            wandb.log({'val_acc': val_acc, 'model_size_bytes': size_info['total_bytes'],
                       'act_bytes_quanted': act_bytes_quant, 'act_bytes_fp32': act_bytes_uncompressed})
            # log per-layer as table
            for lname, linfo in size_info['per_layer'].items():
                wandb.log({f"{lname}_bytes": linfo['layer_total_bytes']})
            wandb.finish()
        results.append(result)
    return results

# ------------ Eval helper ------------
def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(device); y = y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

# ------------ main CLI orchestrator ------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MobileNetV2 CIFAR-10 Deep Compression')
    parser.add_argument('--stage', choices=['prune','quantize','sweep','profile_activations'], required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--sparsity', type=float, default=0.8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--out', type=str, default='compressed_out')
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
    args = parser.parse_args()

    class Cfg:
        data_dir = args.data_dir
        batch_size = args.batch_size
        num_workers = args.workers

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = GetCifar10(Cfg) 
    
    
    model = MobileNetV2(num_classes=10, width_mult=args.width_mult, dropout=args.dropout)
    model = model.to(device)


    if args.stage == 'prune':
        assert args.ckpt, "provide baseline ckpt"
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        mask, thr = make_global_mag_prune_mask(model, args.sparsity)
        apply_mask_to_model(model, mask)
        torch.save({'state': model.state_dict(), 'mask': {k: v.cpu().numpy() for k,v in mask.items()}}, '/content/drive/MyDrive/deepCompression_ckpts/pruned_ckpt.pth')
        print("Saved pruned checkpoint pruned_ckpt.pth")

    elif args.stage == 'quantize':
        # expects pruned_ckpt.pth with 'mask' and 'state'
        assert args.ckpt, "provide pruned ckpt"
        package = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(package['state'])
        mask_np = package.get('mask', {})
        mask = {k: torch.from_numpy(v).to(device) for k,v in mask_np.items()}
        bits_map = {'classifier':5, 'features':8}
        meta = build_weight_sharing_metadata(model, mask, bits_map)
        # save meta
        np_meta = {}
        for k,v in meta.items():
            np_meta[k + '__cb'] = v['codebook']
            np_meta[k + '__idx'] = v['indices']
        np.savez('quant_meta_initial.npz', **np_meta)
        # centroid registry and QAT
        reg = CentroidRegistry(meta, mask, device=device)
        apply_centroids_to_model(model, reg)
        centroid_params = [getattr(reg, reg.layers[n]['cent_param_name']) for n in reg.layers]
        optim = torch.optim.SGD(centroid_params, lr=1e-2, momentum=0.9)
        #fine-tune with train loader
        for epoch in range(10):  # adjust the number of epochs as needed
            model.train()
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                optim.zero_grad()
                # Apply centroids before forward
                apply_centroids_to_model(model, reg)
                outputs = model(imgs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optim.step()
        # after centroid finetune:
        apply_centroids_to_model(model, reg)
        # save final quantized model and meta
        torch.save({'state': model.state_dict()}, '/content/drive/MyDrive/deepCompression_ckpts/quantized_model.pth')
        np.savez('quant_meta_final.npz', **np_meta)
        print("Saved quantized_model.pth and quant_meta_final.npz")

    elif args.stage == 'sweep':
        # Demonstration: create a small sweep grid
        sweep = [
            {'sparsity': 0.6, 'bits_conv':8, 'bits_fc':5, 'centroid_finetune_epochs':5, 'prune_finetune_epochs':2},
            {'sparsity': 0.8, 'bits_conv':8, 'bits_fc':5, 'centroid_finetune_epochs':10, 'prune_finetune_epochs':5},
            {'sparsity': 0.9, 'bits_conv':8, 'bits_fc':4, 'centroid_finetune_epochs':15, 'prune_finetune_epochs':5},
        ]
        # map config into required items
        configs = []
        for s in sweep:
            cfg = {'sparsity': s['sparsity'], 'bits_map': {'features': s['bits_conv'], 'classifier': s['bits_fc']},
                   'centroid_finetune_epochs': s['centroid_finetune_epochs'], 'prune_finetune_epochs': s['prune_finetune_epochs'],
                   'centroid_lr': 1e-2, 'prune_lr': 1e-2, 'include_mask': True, 'act_bits': 8,
                   'index_diff_bits_map': {'features':8, 'classifier':5}, 'act_batches': 5}
            configs.append(cfg)
        results = run_sweep_and_log(model_fn=model, dataloaders=(train_loader, test_loader),
                                   device=device, sweep_configs=configs, out_dir='/content/drive/MyDrive/deepCompression_ckpts/results_sweep/', use_wandb=args.use_wandb)
        with open('/content/drive/MyDrive/deepCompression_ckpts/sweep_results.json','w') as f:
            json.dump(results, f, default=lambda o: o if isinstance(o,(int,float,str)) else str(o))
        print("Sweep done, results saved to /content/drive/MyDrive/deepCompression_ckpts/sweep_results.json")

    elif args.stage == 'profile_activations':
        # sample activation profiling
        if args.ckpt:
            model.load_state_dict(torch.load(args.ckpt, map_location=device))
        act_col = ActivationStatsCollector(model)
        act_col.add_hooks()
        with torch.no_grad():
            # run a few batches
            for i, (x,y) in enumerate(test_loader):
                _ = model(x.to(device))
                if i >= 10: break
        act_col.remove_hooks()
        print("Activation avg bytes per sample per monitored layer:", act_col.summary())
    else:
        print("Unknown stage")#
