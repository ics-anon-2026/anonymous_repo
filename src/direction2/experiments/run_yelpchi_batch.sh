#!/bin/bash
# 批量跑 YelpChi BinaryGAT 关键方法，5 个 seed

set -e

cd /d/workbuddy工作区/SCI论文发表
PY="d:/Miniconda3/python.exe"
BASE="$PY src/direction2/experiments/run_ccs.py --dataset YelpChi --model BinaryGAT --epochs 100 --device cpu --use_val_threshold --lr 0.001 --dropout 0.1 --balanced_batch --pos_weight 1.0"

for seed in 0 1 2 3 4; do
    echo "=== YelpChi random seed=$seed ==="
    $BASE --method random --topk 20 --seed $seed
done

for seed in 0 1 2 3 4; do
    echo "=== YelpChi RIS(noSim) seed=$seed ==="
    $BASE --method ris --topk 20 --seed $seed --importance_mode relation_sim --lambda_stab 1.0 --alpha 1.0 --no_sim
done

for seed in 0 1 2 3 4; do
    echo "=== YelpChi CCS seed=$seed ==="
    $BASE --method ccs --topk 20 --seed $seed --n_envs 3 --collab_bonus 0.1
done

echo "=== YelpChi batch done ==="
