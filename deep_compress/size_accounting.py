"""
Functions to compute bytes used by codebooks, indices and sparse metadata.
"""
import numpy as np

def bytes_for_codebook(cb):
    return cb.size * 4

def bytes_for_indices(indices, bitwidth):
    flat = indices.flatten()
    valid = flat[flat >= 0]
    total_bits = valid.size * bitwidth
    return (total_bits + 7) // 8

def bytes_for_index_meta(indices, diff_bits=8):
    flat = indices.flatten()
    nz = np.where(flat>=0)[0]
    return (nz.size * diff_bits + 7)//8

def compute_model_size(meta, include_mask=False, mask_bits_per_param=1, index_diff_map=None):
    total = 0
    per_layer = {}
    for name, info in meta.items():
        cb = info['codebook']; idx = info['indices']; b = info['bitwidth']
        cb_b = bytes_for_codebook(cb)
        idx_b = bytes_for_indices(idx, b)
        diff_b = 8
        if index_diff_map:
            for s,bits in index_diff_map.items():
                if s in name:
                    diff_b = bits; break
        meta_b = bytes_for_index_meta(idx, diff_b)
        mask_b = 0
        if include_mask:
            mask_b = (idx.size * mask_bits_per_param + 7)//8
        layer_total = cb_b + idx_b + meta_b + mask_b
        per_layer[name] = {'codebook_bytes':cb_b, 'indices_bytes':idx_b, 'meta_bytes':meta_b, 'mask_bytes':mask_b, 'layer_total':layer_total}
        total += layer_total
    return {'total_bytes': total, 'per_layer': per_layer}
