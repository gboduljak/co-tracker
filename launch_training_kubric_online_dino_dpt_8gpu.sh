#!/bin/bash

EXP_DIR="/scratch/shared/beegfs/gabrijel/experiments/v-jepa-probing/cotracker"
EXP_NAME="dpt-dino"
DATE="13-08-2025"
DATASET_ROOT="/scratch/shared/beegfs/gabrijel/benchmarks"
NUM_STEPS=200000

echo `which python`

mkdir -p ${EXP_DIR}/${DATE}_${EXP_NAME}/logs/
mkdir -p ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3
find . \( -name "*.sh" -o -name "*.py" \) -type f -exec cp --parents {} ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3 \;

export PYTHONPATH=`(cd ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3 && pwd)`:`pwd`:$PYTHONPATH
# Find available port
export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export WORLD_SIZE=8
export NODES=1

# Debug info
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Allocated nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"

srun \
  --nodes=$NODES \
  --ntasks=8 \
  --ntasks-per-node=8 \
  --gpus-per-node=8 \
  --export=ALL \
  python ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3/train_on_kubric.py \
    --batch_size 1 \
    --accumulation_steps 4 \
    --num_steps ${NUM_STEPS} \
    --ckpt_path ${EXP_DIR}/${DATE}_${EXP_NAME} \
    --model_name cotracker_three_dino \
    --save_freq 512 \
    --sequence_len 64 \
    --eval_datasets tapvid_davis_first tapvid_stacking \
    --traj_per_sample 384 \
    --sliding_window_len 16 \
    --train_datasets kubric \
    --save_every_n_epoch 2 \
    --evaluate_every_n_epoch 2 \
    --model_stride 4 \
    --dataset_root ${DATASET_ROOT} \
    --num_virtual_tracks 64 \
    --mixed_precision \
    --corr_radius 3 \
    --wdecay 0.0005 \
    --linear_layer_for_vis_conf \
    --validate_at_start \
    --add_huber_loss \
    --log_every_n_steps 32 \
    --num_nodes $NODES \
    --gradient_checkpointing \
    --upsampling_type "dpt"    
  2>&1 | tee ${EXP_DIR}/${DATE}_${EXP_NAME}/logs/train.log