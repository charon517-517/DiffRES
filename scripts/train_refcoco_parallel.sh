#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export CUDA_VISIBLE_DEVICES=1,2,3
#   bash scripts/train_refcoco_parallel.sh [DATA_DIR] [BS_PER_GPU] [TRAIN_STEPS]

export CUDA_VISIBLE_DEVICES=1,2,3
DATA_DIR="${1:-/root/autodl-tmp/refer_data}"
BS_PER_GPU="${2:-1}"
TRAIN_STEPS="${3:-35000}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

# Optional VAE ckpt (NOT shipped). If you have it, set:
#   export VAE_CKPT=/path/to/model.pt
VAE_CKPT="${VAE_CKPT:-}"
VAE_ARG=()
if [[ -n "$VAE_CKPT" ]]; then
  VAE_ARG+=("base.vae_model_kwargs.pretrained_path=$VAE_CKPT")
  VAE_ARG+=("base.load_vae=True")
else
  VAE_ARG+=("base.load_vae=False")
  VAE_ARG+=("base.model_kwargs.in_channels=4")
  VAE_ARG+=("base.vae_model_kwargs.in_channels=1")
fi

python -W ignore main_ldm.py  datasets=refcoco  env.data_dir="$DATA_DIR"  base.wandb=False  base.eval_only=False  base.train_db_name=refcoco  base.val_db_name=refcoco  base.refcoco_split=val  base.train_kwargs.batch_size=$BS_PER_GPU  base.train_kwargs.train_num_steps=$TRAIN_STEPS  base.train_kwargs.num_workers=0 base.train_kwargs.find_unused_parameters=True  base.eval_kwargs.num_workers=0  base.eval_kwargs.vis_every=999999 base.eval_kwargs.metrics=none   base.eval_kwargs.print_freq=50  base.eval_kwargs.batch_size=1  base.train_kwargs.one_step=True  base.train_kwargs.no_noise_input=True  base.train_kwargs.text_adapter=True  base.train_kwargs.adapter_type=selfattn  base.noise_scheduler_kwargs.prediction_type=sample  base.train_kwargs.min_noise_level=999  "${VAE_ARG[@]}"
