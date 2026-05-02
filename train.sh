#!/bin/bash
set -e

GPU=${1:-0}
EXP=${2:-0}

# Optional: set your wandb API key here if not already logged in

WANDB_PROJECT="UDCHealth"
WANDB_GROUP=""
WANDB_NAME="UDCHealth"
export WANDB_API_KEY="your_key_here"

cd "$(dirname "$0")/src"

echo "============================================================"
echo " UDCHealth training  |  dataset=CUSTOM  gpu=$GPU  exp=$EXP"
echo "============================================================"

# ── Step 0: preprocess & cache ────────────────────────────────
# Converts records_final.pkl → datasets_pre_stand.pkl under data/REC/CUSTOM/processed/.
# Exits immediately with "Please run again!" on first call; no-op if cache already exists.
CACHE=../data/REC/CUSTOM/processed/datasets_pre_stand.pkl
if [ ! -f "$CACHE" ]; then
    echo ""
    echo "=== [0/3] Preprocessing data (first-time cache) ==="
    python main_rec.py --exp_num "$EXP" --gpu "$GPU"
    echo "Cache saved. Continuing to training..."
fi

# ── Step 1+2: pretrain PCF + DRL ──────────────────────────────
# PCF  – trains the backbone EHR model (Transformer) on full visit sequences.
# DRL  – aligns PCF embeddings with Sap-BERT via cross-modal vector quantization.
# Both checkpoints are saved to log/ckpt/REC/CUSTOM-Transformer-<exp>/.
echo ""
echo "=== [1+2/3] Pretrain PCF + DRL ==="
python main_rec.py --pretrain --exp_num "$EXP" --gpu "$GPU" \
    --use_wandb --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_GROUP" \
    --wandb_name "pretrain-exp${EXP}"

# ── Step 3: aug inference (two thresholds) ────────────────────
# Loads saved PCF + DRL checkpoints, refines rare-condition embeddings,
# ensembles with original PCF output, and reports final metrics.
# Results are written to separate subdirs per threshold:
#   log/ckpt/REC/CUSTOM-Transformer-udc-<EXP>_t04/
#   log/ckpt/REC/CUSTOM-Transformer-udc-<EXP>_t02/
echo ""
echo "=== [3/3a] Aug Inference + Tuning  threshold=0.4  (fine-tunes once) ==="
python main_rec.py --tuning --exp_num "$EXP" --gpu "$GPU" --thres 0.4 \
    --use_wandb --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_GROUP" \
    --wandb_name "tuning-exp${EXP}-t04"

echo ""
echo "=== [3/3b] Aug Inference  threshold=0.2  (reuses tuned checkpoint) ==="
python main_rec.py --exp_num "$EXP" --gpu "$GPU" --thres 0.2 \
    --use_wandb --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_GROUP" \
    --wandb_name "infer-exp${EXP}-t02"

echo ""
echo "All done!"
