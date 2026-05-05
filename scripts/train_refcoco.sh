#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/train_refcoco.sh [DATA_DIR] [BS]
DATA_DIR="${1:-/root/autodl-tmp/refer_data}"
BS="${2:-8}"

# Select GPUs externally (recommended):
#   export CUDA_VISIBLE_DEVICES=1,2,3
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

# Optional VAE ckpt (not shipped in this repo)
VAE_CKPT="${VAE_CKPT:-}"
VAE_ARG=()
if [[ -n "$VAE_CKPT" ]]; then
  VAE_ARG+=("base.vae_model_kwargs.pretrained_path=$VAE_CKPT")
  VAE_ARG+=("base.load_vae=True")
else
  # If you don't have the VAE ckpt, you may need to set base.load_vae=False and adjust configs.
  VAE_ARG+=("base.load_vae=False")
fi

python -W ignore main_ldm.py  datasets=refcoco  env.data_dir="$DATA_DIR"  base.train_db_name=refcoco  base.val_db_name=refcoco  base.refcoco_split=val  base.train_kwargs.batch_size=$BS  base.train_kwargs.num_workers=0  base.eval_kwargs.num_workers=0  base.train_kwargs.one_step=True  base.train_kwargs.no_noise_input=True  base.train_kwargs.text_adapter=True  base.train_kwargs.adapter_type=selfattn  base.noise_scheduler_kwargs.prediction_type=sample  base.train_kwargs.min_noise_level=999  base.eval_kwargs.metrics=miou  base.eval_kwargs.batch_size=1  "${VAE_ARG[@]}"
