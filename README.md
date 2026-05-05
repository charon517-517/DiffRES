# DiffRES

Official code release for **DiffRES: Unleashing Text-to-Image Diffusion Models for Generative Referring Expression Segmentation without Information Leakage**.

This repository is a **cleaned & minimal** version extracted from our internal codebase. It keeps only the core components for **training / evaluation / inference**.

## Features
- Stable Diffusion v1.5 backbone for generative referring expression segmentation (RES).
- **One-step mask generation** (no iterative denoising at inference).
- Text-conditioning via a lightweight adapter (configurable).

## Installation
We recommend using conda (see `environment.yaml`). A minimal pip install list is also provided in `requirements.txt`.

```bash
cd DiffRES
# conda env create -f environment.yaml
# conda activate diffres

pip install -r requirements.txt
```

## Prepare assets
### 1) Stable Diffusion v1.5
Set `base.pretrained_model_path` (default in `configs/base/base.yaml`). Example path used on our server:

- `/root/autodl-fs/weights/stable-diffusion-v1-5`

### 2) RefCOCO / RefCOCO+ / RefCOCOg
Put the datasets under a folder (example):

- `DATA_DIR=/root/autodl-tmp/refer_data`

and pass `env.data_dir=$DATA_DIR` when running.

### 3) (Optional) VAE checkpoint
Some scripts reference a VAE checkpoint:

- `base.vae_model_kwargs.pretrained_path=/path/to/model.pt`

This file is **NOT included** in this repo.

## Quick start
All commands are launched via `main_ldm.py` (Hydra configs in `configs/`).

### Training
```bash
cd /root/DiffRES
export CUDA_VISIBLE_DEVICES=1,2,3

bash scripts/train_refcoco.sh
```

### Evaluation
```bash
cd /root/DiffRES
export CUDA_VISIBLE_DEVICES=1

CKPT=/path/to/checkpoint.pt
bash scripts/eval_refcoco.sh $CKPT
```

## Structure
- `main_ldm.py`: entry point
- `configs/`: configs
- `ldmseg/`: core code (model/trainer/dataset/utils)
- `scripts/`: runnable scripts

## License
TBD
