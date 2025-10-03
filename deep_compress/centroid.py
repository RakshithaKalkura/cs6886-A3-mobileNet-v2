"""

Centroid registry used for centroid fine-tuning (QAT). Register centroids as Parameters
and reconstruct weights. This version is robust to mask entries being torch tensors
(on CPU or CUDA), numpy arrays, or Python lists.

API:
    registry = CentroidRegistry(meta, mask_dict, device='cuda')
    apply_centroids_to_model(model, registry)
"""

import numpy as np
import torch
import torch.nn as nn

class CentroidRegistry(nn.Module):
    def __init__(self, meta, mask_dict, device='cpu'):
        """
        meta: dict mapping layer_name -> { 'codebook': np.array([...]), 'indices': np.array([...]), 'shape': tuple(...) }
        mask_dict: dict mapping layer_name -> mask (torch.Tensor on CPU/CUDA or numpy array or list)
        device: device string or torch.device where centroids/buffers should live
        """
        super().__init__()
        self.device = torch.device(device)
        self.layers = {}

        for name, info in meta.items():
            codebook = info.get('codebook', np.array([], dtype=np.float32))
            indices = info.get('indices', np.array([], dtype=np.int32))
            shape = tuple(info.get('shape', ()))

            # Normalize mask: accept torch.Tensor, numpy array, or list
            raw_mask = None
            if isinstance(mask_dict, dict):
                raw_mask = mask_dict.get(name, None)

            if raw_mask is None:
                mask_np = np.ones(shape, dtype=np.float32)
            else:
                if isinstance(raw_mask, torch.Tensor):
                    # move to CPU and convert to numpy
                    mask_np = raw_mask.detach().cpu().numpy().astype(np.float32)
                else:
                    # assume numpy-like (ndarray or list)
                    mask_np = np.array(raw_mask).astype(np.float32)

                # ensure shape matches
                if mask_np.shape != shape:
                    # try to reshape or broadcast if possible; otherwise fall back to ones
                    try:
                        mask_np = mask_np.reshape(shape)
                    except Exception:
                        mask_np = np.ones(shape, dtype=np.float32)

            # Create centroids parameter (may be empty)
            if codebook is None:
                codebook = np.array([], dtype=np.float32)
            if codebook.size > 0:
                cent_param = nn.Parameter(torch.tensor(codebook.astype(np.float32), device=self.device))
            else:
                cent_param = nn.Parameter(torch.zeros(0, dtype=torch.float32, device=self.device))

            # register parameter with a unique name based on layer
            param_name = name.replace('.', '_') + '_cent'
            # buffer names for indices and mask
            idx_buf_name = name.replace('.', '_') + '_idx'
            mask_buf_name = name.replace('.', '_') + '_mask'

            # Convert indices and mask to tensors on the chosen device
            # indices should be int32 on device (we keep indices on device to reconstruct easily)
            if isinstance(indices, torch.Tensor):
                idx_tensor = indices.detach().to(self.device).to(torch.int32)
            else:
                idx_tensor = torch.from_numpy(np.asarray(indices, dtype=np.int32)).to(self.device)

            mask_tensor = torch.from_numpy(np.asarray(mask_np, dtype=np.float32)).to(self.device)

            # register param/buffers
            # NOTE: register_parameter must use a valid Python identifier name
            # Use setattr/register_parameter to attach parameter with param_name
            self.register_parameter(param_name, cent_param)
            # register buffers
            self.register_buffer(idx_buf_name, idx_tensor)
            self.register_buffer(mask_buf_name, mask_tensor)

            # store layer bookkeeping
            self.layers[name] = {
                'cent_name': param_name,
                'idx_name': idx_buf_name,
                'mask_name': mask_buf_name,
                'shape': shape
            }

    def reconstruct_param(self, name):
        """
        Reconstruct a full-weight tensor for layer `name` using centroids, indices and mask.
        Returns a torch tensor on self.device with shape self.layers[name]['shape'].
        """
        info = self.layers[name]
        cent = getattr(self, info['cent_name'])
        idx = getattr(self, info['idx_name'])
        mask = getattr(self, info['mask_name'])

        # zero-filled fallback if no centroids
        if cent.numel() == 0:
            return torch.zeros(info['shape'], dtype=torch.float32, device=self.device)

        # flatten indices, safe-index into centroids (replace -1 with 0)
        idx_flat = idx.view(-1)
        safe_idx = idx_flat.clone()
        safe_idx[safe_idx < 0] = 0  # map pruned entries to centroid 0 (mask will zero them anyway)

        # gather centroid values and reshape
        gathered = cent[safe_idx].view(info['shape'])
        # apply mask so pruned entries are exactly zero
        reconstructed = gathered * mask
        return reconstructed

def apply_centroids_to_model(model, centroid_registry):
    """
    Overwrite model weight tensors (in-place) with reconstructed quantized weights
    from centroid_registry. Only updates parameters whose names appear in centroid_registry.layers.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'weight' in name and name in centroid_registry.layers:
                new_w = centroid_registry.reconstruct_param(name).to(p.device)
                p.data.copy_(new_w)
