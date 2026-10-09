#!/bin/bash
# Amazon CCS ablation study (5 seeds)
set -e
PY="d:/Miniconda3/python.exe"
RUN="src/direction2/experiments/run_ccs.py"
COMMON="--dataset Amazon --model BinaryGAT --topk 20 --epochs 100 --device cpu --use_val_threshold --lr 0.001 --dropout 0.1 --balanced_batch --pos_weight 1.0 --n_envs 3 --collab_bonus 0.1"

for seed in 0 1 2 3 4; do
    echo "=== Amazon CCS full seed=$seed ==="
    $PY $RUN $COMMON --method ccs --seed $seed

    echo "=== Amazon CCS no_stab seed=$seed ==="
    $PY $RUN $COMMON --method ccs --seed $seed --no_stab

    echo "=== Amazon CCS no_collab seed=$seed ==="
    $PY $RUN $COMMON --method ccs --seed $seed --no_collab

    echo "=== Amazon CCS only_sim seed=$seed ==="
    $PY $RUN $COMMON --method ccs --seed $seed --no_stab --no_collab
done

echo "=== Amazon CCS ablation done ==="
