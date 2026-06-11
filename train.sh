#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
PYTHON_BIN=${PYTHON_BIN:-/home/zly/miniconda3/envs/OSMTrack/bin/python}
CONFIG=${CONFIG:-TrackingmambaV2-ep150-full-256}
SAVE_DIR=${SAVE_DIR:-./output}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE}" \
  lib/train/run_training.py \
  --script trackingmambav2 \
  --config "${CONFIG}" \
  --save_dir "${SAVE_DIR}" \
  --use_lmdb 0
