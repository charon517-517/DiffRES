#!/usr/bin/env bash
set -euo pipefail

# Simple "inference" entry: run evaluation loop on RefCOCO split using a checkpoint.
# Usage: bash scripts/infer_refcoco_eval.sh /path/to/ckpt.pt [DATA_DIR]

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/infer_refcoco_eval.sh /path/to/ckpt.pt [DATA_DIR]"
  exit 1
fi

CKPT="$1"
DATA_DIR="${2:-/root/autodl-tmp/refer_data}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

python -W ignore main_ldm.py  datasets=refcoco  env.data_dir="$DATA_DIR"  base.wandb=False  base.eval_only=True  base.load_path="$CKPT"  base.load_vae=False  base.train_db_name=refcoco  base.val_db_name=refcoco  base.refcoco_split=val  base.eval_kwargs.batch_size=1  base.eval_kwargs.num_workers=0  base.eval_kwargs.print_freq=10  base.eval_kwargs.threshold_output=False  base.eval_kwargs.metrics=miou
