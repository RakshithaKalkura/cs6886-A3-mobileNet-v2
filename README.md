# MobileNet-v2 Quantisation (Assignment 3 for cs6886)




## Deep Compression Pipeline
 Deep Compression Modular Codebase

Files:
 - model_factory.py : returns MobileNetV2 adapted for CIFAR-10. Replace or adapt to your own model.
 - train.py         : training loop (baseline training). Uses user's dataloader through import.
 - evaluate.py      : evaluation helper (validation/test).
 - compress/        : pruning, quantization, centroid fine-tuning, Huffman and size accounting modules.
 - utils.py         : helper utilities for saving/loading checkpoints and masks.
 - sweep.py         : orchestrates pruning+quantize sweeps and logs results (wandb optional).

Usage (specify checkpoints appropriately):
 1) Train baseline: 
 ``` 
 python train.py --epochs 60 --out ckpt_baseline.pth
 ```
 2) Prune: 
  ```
  python compress/prune.py --ckpt ckpt_baseline.pth --sparsity 0.8 --out pruned_ckpt.pth
   ```
 3) Quantize & centroid finetune: 
  ```
  python compress/quantize.py --ckpt pruned_ckpt.pth --out quantized_meta.npz
   ```
 4) Huffman: 
  ```
  python compress/huffman.py --meta quantized_meta.npz --out compressed_package.npz
   ```
 5) Sweep: 
 ```
  python sweep.py --cfgs sweep_configs.json
   ```
