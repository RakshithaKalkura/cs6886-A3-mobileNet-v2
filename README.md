# MobileNet-v2 Compression

This work was done as an assignment for Systems for Deep Learning course (CS6886)

This repository provides a clean, modular PyTorch implementation of the Deep Compression pipeline — combining pruning → quantization → Huffman coding — applied to MobileNetV2 trained on CIFAR-10.

It supports iterative pruning with fine-tuning, centroid quantisation with QAT, activation profiling, Huffman packaging, and storage overhead analysis.


## Deep Compression Pipeline
### Deep Compression Modular Codebase - Project sturcture
├── deep_compress/                # Core compression modules
│   ├── prune.py             # Global magnitude pruning (single-shot)
│   ├── quantize.py          # Quantisation and metadata generation
│   ├── centroids.py         # Centroid registry and fine-tuning (QAT)
│   ├── huffman.py           # Huffman coding utilities
│   └── size_accounting.py   # Model size analysis
├── dataset/                 # CIFAR-10 dataloader wrapper
├── models/                  # MobileNetV2 adapted for CIFAR-10
├── train.py                 # Baseline training script
├── evaluate.py              # Accuracy evaluation helper
├── sweep.py                 # Main sweep driver (iterative pruning + compression)
├── utils.py                 # Checkpoint + device helpers
├── analyse_results.py       # Analyse sweep and get the best compression ratio
├── plot_parallel_wandb.py   # Plotting the wandb parallel coordinates plot
├── requirements.txt
└── README.md


## Installation 
```
git clone https://RakshithaKalkura/cs6886-A3-mobileNet-v2.git
cd cs6886-A3-mobileNet-v2
pip install -r requirements.txt
```
The hardware and software setup are briefly discussed in the report. Please refer to that. TLDR; A100 GPUs were used for the training and experiments.

## Usage (specify checkpoints appropriately):
 1) Train baseline: 
 ``` 
 python train.py --epochs 300 --batch_size 128 --lr 0.1 --out checkpoints/model_best.pth
 ```
You can use the pre-trained model from here - [link](https://drive.google.com/drive/folders/1OqtBDqrzRlmrvljGaIBzPc9YJflb2ieh?usp=sharing)
All the configuration and seed numbers are specified within the code and in the report.

 2) Sweep (Run compression sweep): 
 ```
  python sweep.py --cfgs sweep_configs.json
   ```
