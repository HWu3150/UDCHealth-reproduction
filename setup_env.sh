#!/bin/bash
# Environment setup for UDCHealth on aarch64 + CUDA 13.0
# Assumes conda env already created and activated (Python 3.10)
set -e

echo "=== [1/6] numpy < 2.0 (binary compat fix) ==="
pip install "numpy<2.0" --force-reinstall

echo ""
echo "=== [2/6] PyTorch (aarch64, nightly cu128 — required for Blackwell sm_121) ==="
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

echo ""
echo "=== [3/6] transformers ==="
pip install transformers==4.44.2

echo ""
echo "=== [4/6] pyhealth ==="
pip install pyhealth

echo ""
echo "=== [5/6] Core deps ==="
pip install \
    pandas \
    scikit-learn \
    scipy \
    matplotlib \
    tqdm \
    dill \
    wandb

echo ""
echo "=== [6/6] Verify ==="
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
import numpy as np
print("numpy:", np.__version__)
import transformers
print("transformers:", transformers.__version__)
import pyhealth
print("pyhealth: ok")
print("All good!")
EOF
