#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval_refcoco.sh /path/to/checkpoint.pt [DATA_DIR]"
  exit 1
fi

CKPT="$1"
DATA_DIR="${2:-/root/autodl-tmp/refer_data}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

python -W ignore main_ldm.py  datasets=refcoco  env.data_dir="$DATA_DIR"  base.eval_only=True  base.load_path="$CKPT"  base.load_vae=False  base.train_db_name=refcoco  base.val_db_name=refcoco  base.refcoco_split=val  base.eval_kwargs.batch_size=1
