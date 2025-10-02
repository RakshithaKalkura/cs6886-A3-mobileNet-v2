
"""
Deep Compression Modular Codebase

Files:
 - model_factory.py : returns MobileNetV2 adapted for CIFAR-10. Replace or adapt to your own model.
 - train.py         : training loop (baseline training). Uses user's dataloader through import.
 - evaluate.py      : evaluation helper (validation/test).
 - compress/        : pruning, quantization, centroid fine-tuning, Huffman and size accounting modules.
 - utils.py         : helper utilities for saving/loading checkpoints and masks.
 - sweep.py         : orchestrates pruning+quantize sweeps and logs results (wandb optional).

Usage (example):
 1) Train baseline: python train.py --epochs 60 --out ckpt_baseline.pth
 2) Prune: python compress/prune.py --ckpt ckpt_baseline.pth --sparsity 0.8 --out pruned_ckpt.pth
 3) Quantize & centroid finetune: python compress/quantize.py --ckpt pruned_ckpt.pth --out quantized_meta.npz
 4) Huffman: python compress/huffman.py --meta quantized_meta.npz --out compressed_package.npz
 5) Sweep: python sweep.py --cfgs sweep_configs.json

"""


# ==== FILE: compress/huffman.py ====
"""
Huffman coding utilities for final packaging. Provides functions to build Huffman codes,
encode indices into a bitstream per-layer, and save a compressed package (NPZ + codes).
This updated module also saves a checkpoint (.npz) that contains:
 - per-layer Huffman codes (as JSON-serializable dicts)
 - per-layer encoded bitstreams (as bytes objects)
 - original codebooks and indices metadata for reconstruction

This file also exposes a simple CLI: run this module with --meta <quant_meta.npz>
and it will produce <out_prefix>_meta.npz and <out_prefix>_codes.json.
"""
from collections import Counter
import heapq
import json
import numpy as np
import os
import argparse

class HuffmanNode:
    def __init__(self, freq, sym=None, left=None, right=None):
        self.freq = freq; self.sym = sym; self.left = left; self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_codes(symbols):
    """Return dict: symbol -> bitstring"""
    freq = Counter(symbols)
    heap = [HuffmanNode(f, s) for s,f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 0:
        return {}
    if len(heap) == 1:
        return {heap[0].sym: '0'}
    while len(heap) > 1:
        a = heapq.heappop(heap); b = heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(a.freq + b.freq, None, a, b))
    root = heap[0]
    codes = {}
    def walk(node, prefix=''):
        if node.sym is not None:
            codes[node.sym] = prefix
        else:
            walk(node.left, prefix + '0')
            walk(node.right, prefix + '1')
    walk(root)
    return codes

def encode_symbols_with_codes(symbols, codes):
    """Encode an iterable of symbols into a bytes object using the provided Huffman codes.
    Returns (bytearray_bytes, bit_length)
    """
    # Build bitstring incrementally to avoid huge memory on extremely large models
    bits = []
    total_bits = 0
    for s in symbols:
        code = codes.get(s)
        if code is None:
            # this can happen if there is only one unique symbol previously; treat as '0'
            code = '0'
        bits.append(code)
        total_bits += len(code)
    bitstr = ''.join(bits)
    # pack into bytes
    b = bytearray()
    for i in range(0, len(bitstr), 8):
        chunk = bitstr[i:i+8]
        if len(chunk) < 8:
            chunk = chunk.ljust(8, '0')
        b.append(int(chunk, 2))
    return bytes(b), total_bits

def encode_indices_layerwise(meta):
    """Given quantization metadata dict (layer -> {'indices': numpy array, ...}),
    produce per-layer Huffman codes and encoded bytes. Returns dicts.
    """
    codes_per_layer = {}
    bitstreams = {}
    bit_lengths = {}
    for name, info in meta.items():
        idx = info.get('indices')
        if idx is None:
            continue
        flat = idx.flatten().tolist()
        if len(flat) == 0:
            codes_per_layer[name] = {}
            bitstreams[name] = np.array([], dtype=np.uint8)
            bit_lengths[name] = 0
            continue
        codes = build_huffman_codes(flat)
        encoded_bytes, bitlen = encode_symbols_with_codes(flat, codes)
        codes_per_layer[name] = {str(k): v for k,v in codes.items()}  # keys as strings for JSON
        bitstreams[name] = np.frombuffer(encoded_bytes, dtype=np.uint8)
        bit_lengths[name] = int(bitlen)
    return codes_per_layer, bitstreams, bit_lengths

def save_huffman_package(meta, out_prefix='compressed_huffman'):
    """Save Huffman-coded package.
    Writes: <out_prefix>_meta.npz containing per-layer:
       - codebook (cb)
       - original indices (idx)
       - encoded_bitstream (as uint8 array)
       - encoded_bit_length (int)
    Also writes <out_prefix>_codes.json with Huffman code maps (string keys).
    Returns paths saved.
    """
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)
    codes, bitstreams, bit_lengths = encode_indices_layerwise(meta)
    npz_dict = {}
    for name, info in meta.items():
        cb = info.get('codebook', np.array([], dtype=np.float32))
        idx = info.get('indices', np.array([], dtype=np.int32))
        bs = bitstreams.get(name, np.array([], dtype=np.uint8))
        bl = bit_lengths.get(name, 0)
        npz_dict[name + '__cb'] = np.asarray(cb, dtype=np.float32)
        npz_dict[name + '__idx'] = np.asarray(idx, dtype=np.int32)
        npz_dict[name + '__bits'] = np.int64(bl)
        npz_dict[name + '__bs'] = np.asarray(bs, dtype=np.uint8)
    npz_path = out_prefix + '_meta.npz'
    np.savez(npz_path, **npz_dict)
    codes_path = out_prefix + '_codes.json'
    with open(codes_path, 'w') as f:
        json.dump(codes, f)
    return npz_path, codes_path

def load_meta_npz(path):
    """Load quantization meta saved as .npz (from compress/quantize.py save_meta_npz).
    Returns a dict meta[name] = {'codebook':..., 'indices': ..., ...}
    """
    archive = np.load(path, allow_pickle=True)
    meta = {}
    # keys like '<layer>__cb' and '<layer>__idx'
    names = set(k.rsplit('__',1)[0] for k in archive.files)
    for name in names:
        cb_key = name + '__cb'
        idx_key = name + '__idx'
        cb = archive[cb_key] if cb_key in archive.files else np.array([],dtype=np.float32)
        idx = archive[idx_key] if idx_key in archive.files else np.array([],dtype=np.int32)
        meta[name] = {'codebook': np.asarray(cb, dtype=np.float32), 'indices': np.asarray(idx, dtype=np.int32)}
    return meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', required=True, help='Input quantization meta .npz produced by compress/quantize.py')
    parser.add_argument('--out_prefix', default='/content/drive/MyDrive/deepcompress_ckpt/compressed_huffman', help='Output prefix for compressed package')
    args = parser.parse_args()
    meta = load_meta_npz(args.meta)
    npz_path, codes_path = save_huffman_package(meta, out_prefix=args.out_prefix)
    print('Saved Huffman package:', npz_path, 'and codes:', codes_path)

if __name__ == '__main__':
    main()
