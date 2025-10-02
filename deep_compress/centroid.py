"""
Centroid registry used for centroid fine-tuning (QAT). Register centroids as Parameters and reconstruct weights.
"""
import torch
import torch.nn as nn
import numpy as np

class CentroidRegistry(nn.Module):
    def __init__(self, meta, mask_dict, device='cpu'):
        super().__init__()
        self.device = device
        self.layers = {}
        for name, info in meta.items():
            cb = info['codebook']
            idx = info['indices']
            shape = tuple(info['shape'])
            mask = mask_dict.get(name, (np.ones(shape))).astype(np.float32) if isinstance(mask_dict, dict) else np.ones(shape, dtype=np.float32)
            if cb.size>0:
                cent = nn.Parameter(torch.tensor(cb, dtype=torch.float32, device=device))
            else:
                cent = nn.Parameter(torch.zeros(0, dtype=torch.float32, device=device))
            buf_idx = torch.from_numpy(idx.astype(np.int32)).to(device)
            buf_mask = torch.from_numpy(mask.astype(np.float32)).to(device)
            pname = name.replace('.','_') + '_cent'
            self.register_parameter(pname, cent)
            self.register_buffer(name.replace('.','_') + '_idx', buf_idx)
            self.register_buffer(name.replace('.','_') + '_mask', buf_mask)
            self.layers[name] = {'cent_name':pname, 'idx_name':name.replace('.','_') + '_idx', 'mask_name':name.replace('.','_') + '_mask', 'shape':shape}

    def reconstruct_param(self, name):
        info = self.layers[name]
        cent = getattr(self, info['cent_name'])
        idx_buf = getattr(self, info['idx_name'])
        mask_buf = getattr(self, info['mask_name'])
        if cent.numel()==0:
            return torch.zeros(info['shape'], device=self.device)
        flat_idx = idx_buf.view(-1)
        safe_idx = flat_idx.clone(); safe_idx[safe_idx<0]=0
        w = cent[safe_idx].view(info['shape'])
        w = w * mask_buf
        return w


def apply_centroids_to_model(model, centroid_registry):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'weight' in name and name in centroid_registry.layers:
                neww = centroid_registry.reconstruct_param(name).to(p.device)
                p.data.copy_(neww)