"""
General utilities: checkpoint save/load and device helpers.
"""
import torch
import os

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location)