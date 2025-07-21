#!/bin/bash

MODEL="baseline_online"
CKPT="/scratch/shared/beegfs/gabrijel/hf/cotracker3/${MODEL}.pth"
DATASET_ROOT="/scratch/shared/beegfs/gabrijel/benchmarks/TAP-Vid"
FLASH_ATTENTION="true"
EVALS_DIR="./evals"
GPU_IDX="3"
DATASETS=("tapvid_davis_first" "tapvid_kinetics_first" "tapvid_robotap_first" "tapvid_stacking_first")
SINGLE_POINT="false"

for DATASET in "${DATASETS[@]}"; do
  echo "Running evaluation on ${DATASET}..."

  python ./cotracker/evaluation/evaluate.py \
    --config-name "eval_${DATASET}" \
    exp_dir="${EVALS_DIR}/${MODEL}/flash_attention-${FLASH_ATTENTION}/${DATASET}" \
    dataset_root="${DATASET_ROOT}" \
    checkpoint="${CKPT}" \
    flash_attention="${FLASH_ATTENTION}" \
    gpu_idx="${GPU_IDX}" \
    single_point="${SINGLE_POINT}"
done
