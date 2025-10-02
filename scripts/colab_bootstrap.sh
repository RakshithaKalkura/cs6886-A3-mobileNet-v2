#!/bin/bash
# Usage: bash scripts/colab_bootstrap.sh <repo-url> <drive-out-dir>
REPO_URL=${1:-"https://github.com/RakshithaKalkura/cs6886-A3-mobileNet-v2.git"}
DRIVE_OUT_DIR=${2:-"/content/drive/MyDrive/checkpoints"}

set -e
git clone $REPO_URL repo || (cd repo && git pull)
cd repo
pip install -r requirements.txt
mkdir -p "$DRIVE_OUT_DIR"
echo "Bootstrapped repo and ensured $DRIVE_OUT_DIR exists"
