#!/bin/bash
#SBATCH --job-name=DyG_TGN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

DATASET="$1"
cd $HOME/lab/forked_repo/DyGLib/
module load python/3.8
source $HOME/envs/tsa/bin/activate

python train_link_prediction_snapshot.py \
    --dataset_name "$1" \
    --model_name TGN \
    --num_neighbors 20 \
    --num_layers 1 \
    --num_heads 2 \
    --dropout 0.1 \
    --time_scaling_factor 0.000001 \
    --time_feat_dim 100  \
    --num_runs 1 \
    --gpu 0 \

