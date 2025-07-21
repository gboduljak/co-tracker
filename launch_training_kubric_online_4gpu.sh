#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cotracker

EXP_DIR="/scratch/shared/beegfs/gabrijel/experiments/v-jepa-probing/cotracker"
EXP_NAME="baseline-online"
DATE="20-07-2025"
DATASET_ROOT="/scratch/shared/beegfs/gabrijel/benchmarks"
NUM_STEPS=400000

echo `which python`

mkdir -p ${EXP_DIR}/${DATE}_${EXP_NAME}/logs/
mkdir -p ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3
find . \( -name "*.sh" -o -name "*.py" \) -type f -exec cp --parents {} ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3 \;

export PYTHONPATH=`(cd ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3 && pwd)`:`pwd`:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1,2,3"

python ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3/train_on_kubric.py --batch_size 1 \
--accumulation_steps 8 \
--num_steps ${NUM_STEPS} --ckpt_path ${EXP_DIR}/${DATE}_${EXP_NAME} --model_name cotracker_three \
--save_freq 512 --sequence_len 64 --eval_datasets tapvid_davis_first tapvid_stacking \
--traj_per_sample 384 --sliding_window_len 16 --train_datasets kubric \
--save_every_n_epoch 2 --evaluate_every_n_epoch 2 --model_stride 4 --dataset_root ${DATASET_ROOT} --num_nodes 1 \
--num_virtual_tracks 64 --mixed_precision \
--corr_radius 3 --wdecay 0.0005 --linear_layer_for_vis_conf --validate_at_start --add_huber_loss \
--log_every_n_steps 32 \
--flash_attention \
--restore_ckpt "/scratch/shared/beegfs/gabrijel/experiments/v-jepa-probing/cotracker/20-07-2025_baseline-online/model_cotracker_three_002936.pth"
2>&1 | tee ${EXP_DIR}/${DATE}_${EXP_NAME}/logs/train.log
