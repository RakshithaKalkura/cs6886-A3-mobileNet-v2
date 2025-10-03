# MobileNet-v2 Compression

This work was done as an assignment for Systems for Deep Learning course (CS6886)

This repository provides a clean, modular PyTorch implementation of the Deep Compression pipeline — combining pruning → quantization → Huffman coding — applied to MobileNetV2 trained on CIFAR-10.

It supports iterative pruning with fine-tuning, centroid quantisation with QAT, activation profiling, Huffman packaging, and storage overhead analysis.

---
## Deep Compression Pipeline
### Deep Compression Codebase - Project sturcture
- **`compress/`**: Core compression modules  
  - **`prune.py`**: Global magnitude pruning (single-shot)  
  - **`quantize.py`**: Quantization and metadata generation  
  - **`centroids.py`**: Centroid registry and fine-tuning (QAT)  
  - **`huffman.py`**: Huffman coding utilities  
  - **`size_accounting.py`**: Model size analysis  

- **`dataset/`**: CIFAR-10 dataloader wrapper  
- **`models/`**: MobileNetV2 adapted for CIFAR-10  
- **`train.py`**: Baseline training script  
- **`evaluate.py`**: Accuracy evaluation helper  
- **`sweep.py`**: Main sweep driver (iterative pruning + compression)  
- **`utils.py`**: Checkpoint + device helpers  
- **`requirements.txt`**  
- **`README.md`**

---
## Installation 
```
git clone https://RakshithaKalkura/cs6886-A3-mobileNet-v2.git
cd cs6886-A3-mobileNet-v2
pip install -r requirements.txt
```
The hardware and software setup are briefly discussed in the report. Please refer to that. TLDR; A100 GPUs were used for the training and experiments.
---
## Usage (specify checkpoints appropriately):
- Make a note that you will have to create a data folder to download the CIFAR10 dataset.
- Make necessary and appropriate changes in the code w.r.t directories/path.
 1) Train baseline: 
 ``` 
 python train.py --epochs 300 --batch_size 128 --lr 0.1 --out output_dir
 ```
- You can use the pre-trained model from here - [link](https://drive.google.com/drive/folders/1OqtBDqrzRlmrvljGaIBzPc9YJflb2ieh?usp=sharing). It's also available in the  ckpt folder. 
- All the configuration and seed numbers are specified within the code and in the report.

 2) Sweep (Run compression sweep): 
 ```
  python sweep.py --cfgs sweep_configs.json
   ```
 3) Visulise (WandB plots):
  ```
   python plot_parallel_wandb.py --indir sweep_dir --outdir results_dir --out both --wandb
   ```
 4) Analyse results (optional):
```
python analyse_results.py
```
---

## Outputs

- **Per-run results:**  
  `sweep_out/<run_name>_result.json`  

- **Huffman packages:**  
  `sweep_out/<run_name>_huff_meta.npz`  
  `sweep_out/<run_name>_huff_codes.json`  

- **Sweep summary:**  
  `sweep_summary.json`  

- **W&B logging (if enabled):**  
  - Accuracy, compression ratios, per-layer stats  
  - Parallel coordinates plots for sweep visualization  

Reported metrics include:
- Baseline vs. compressed accuracy  
- Model/weights/activations compression ratios  
- Final model size (MB) after Huffman coding  
- Storage overheads (metadata, codebooks, scaling factors)  

---

## Example Results

- **Baseline accuracy:** ~95.9% on CIFAR-10  
- **Iterative pruning:** up to 70–85% sparsity without accuracy loss  
- **Quantization:** Conv (7–8 bits), FC (5–6 bits), Activations (6–8 bits)  
- **Final compression:**  
  - 4–5× smaller with no accuracy drop  
  - Up to 7–8× smaller with <0.5% accuracy drop  
