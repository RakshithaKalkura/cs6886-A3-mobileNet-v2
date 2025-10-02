"""
Quantization driver: builds k-means clusters per-layer and fine-tunes centroids.
Saves meta as .npz containing codebooks and index arrays.
"""
import argparse
import numpy as np
import torch
from sklearn.cluster import KMeans
from .utils import load_checkpoint, save_checkpoint, get_device
from models import MobileNetV2
from .centroid import CentroidRegistry, apply_centroids_to_model


def cluster_layer(weights_flat, n_clusters):
    if weights_flat.size==0:
        return np.array([],dtype=np.float32), np.array([],dtype=np.int32)
    k = min(n_clusters, weights_flat.size)
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(weights_flat.reshape(-1,1))
    return km.cluster_centers_.squeeze().astype(np.float32), km.labels_.astype(np.int32)

def build_meta(model, mask, bits_map):
    meta = {}
    for name,p in model.named_parameters():
        if 'weight' not in name: continue
        m = mask.get(name, np.ones_like(p.detach().cpu().numpy()))
        flat = p.detach().cpu().numpy().flatten()
        nz = np.where(m.flatten()!=0)[0]
        if nz.size==0:
            meta[name] = {'codebook': np.array([],dtype=np.float32), 'indices': -1*np.ones_like(flat,dtype=np.int32), 'shape': p.shape, 'bitwidth':1, 'nonzeros':0}
            continue
        vals = flat[nz]
        # select bits
        bit = None
        for k_sub,b in bits_map.items():
            if k_sub in name:
                bit=b; break
        if bit is None: bit=8
        k = max(2, 2**bit)
        cb, labels = cluster_layer(vals, k)
        indices = -1*np.ones_like(flat,dtype=np.int32)
        indices[nz] = labels
        meta[name] = {'codebook':cb, 'indices':indices.reshape(p.shape), 'shape':p.shape, 'bitwidth':bit, 'nonzeros':int(nz.size)}
    return meta

def save_meta_npz(meta, path):
    npz_dict = {}
    for name,info in meta.items():
        npz_dict[name+ '__cb'] = info['codebook']
        npz_dict[name+ '__idx'] = info['indices']
    np.savez(path, **npz_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default='/content/drive/MyDrive/deepcompress_ckpt/quant_meta_initial.npz')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--bits_conv', type=int, default=8)
    parser.add_argument('--bits_fc', type=int, default=5)
    args=parser.parse_args()
    device = get_device(args.device)
    model = MobileNetV2().to(device)
    ck = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(ck['state_dict'])
    mask = ck.get('mask', {})
    # bits map: simple substring mapping
    bits_map = {'features': args.bits_conv, 'classifier': args.bits_fc}
    meta = build_meta(model, mask, bits_map)
    save_meta_npz(meta, args.out)
    print('Saved quantization meta to', args.out)

if __name__=='__main__':
    main()