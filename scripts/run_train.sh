#!/bin/bash
# Usage: bash scripts/run_train.sh <epochs> <batch_size>
EPOCHS=${1:-200}
BATCH=${2:-128}
python train.py --data-dir ./data --out-dir /content/drive/MyDrive/mobilenetv2_checkpoints --epochs $EPOCHS --batch-size $BATCH
