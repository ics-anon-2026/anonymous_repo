#!/bin/bash
# YelpChi CCS main comparison (5 seeds)
set -e
PY="d:/Miniconda3/python.exe"
RUN="src/direction2/experiments/run_ccs.py"
COMMON="--dataset YelpChi --model BinaryGAT --topk 20 --epochs 100 --device cpu --use_val_threshold --lr 0.001 --dropout 0.1 --balanced_batch --pos_weight 1.0 --n_envs 3 --collab_bonus 0.1"

for seed in 0 1 2 3 4; do
    echo "=== YelpChi random seed=$seed ==="
    $PY $RUN $COMMON --method random --seed $seed

    echo "=== YelpChi ICS seed=$seed ==="
    $PY $RUN $COMMON --method ccs --seed $seed
done

echo "=== YelpChi CCS batch done ==="
