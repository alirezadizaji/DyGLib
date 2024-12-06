#!/bin/bash
#SBATCH --job-name=DyG_TGAT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

DATASET="$1"
cd $HOME/lab/forked_repo/DyGLib/

python train_link_prediction_snapshot.py \
    --dataset_name "$1" \
    --model_name TGAT \
    --num_neighbors 20 \
    --num_layers 1 \
    --num_heads 2 \
    --dropout 0.1 \
    --time_scaling_factor 0.000001 \
    --time_feat_dim 100  \
    --num_runs 1 \
    --gpu 0

