"""
General utilities: checkpoint save/load and device helpers.
"""
import torch
import numpy as np

def save_checkpoint(state, path):
    import os
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, map_location=None):
    """
    Load a checkpoint robustly:
      - First attempt: normal torch.load (weights_only default)
      - If it errors due to NumPy unpickling restrictions, retry inside
        torch.serialization.safe_globals allowing numpy._core.multiarray._reconstruct.
    SECURITY: only use this for checkpoints you trust.
    """
    try:
        # Normal load (preferred)
        return torch.load(path, map_location=map_location)
    except Exception as e:
        # If error mentions numpy._core.multiarray._reconstruct, retry in safe_globals
        msg = str(e)
        if "numpy._core.multiarray._reconstruct" in msg or "Weights only load failed" in msg:
            try:
                # Use the safe_globals context manager to permit this specific global
                # NOTE: this is safe only for trusted checkpoints.
                with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
                    return torch.load(path, map_location=map_location, weights_only=False)
            except Exception as e2:
                # re-raise with extra context
                raise RuntimeError(
                    "Retry with safe_globals failed. "
                    "If you trust the checkpoint, ensure torch and numpy versions are compatible."
                ) from e2
        # otherwise re-raise original error
        raise

def get_device(pref='cuda'):
    if pref=='cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')