#!/bin/bash
# Amazon CCS ablation after bug fix (5 seeds)
set -e
PY="d:/Miniconda3/python.exe"
RUN="src/direction2/experiments/run_ccs.py"
COMMON="--dataset Amazon --model BinaryGAT --topk 20 --epochs 100 --device cpu --use_val_threshold --lr 0.001 --dropout 0.1 --balanced_batch --pos_weight 1.0 --n_envs 3 --collab_bonus 0.1"

for seed in 0 1 2 3 4; do
    echo "=== Amazon ICS full seed=$seed ==="
    $PY $RUN $COMMON --method ccs --seed $seed

    echo "=== Amazon ICS only_sim seed=$seed ==="
    $PY $RUN $COMMON --method ccs --seed $seed --no_stab --no_collab

    echo "=== Amazon ICS no_stab seed=$seed ==="
    $PY $RUN $COMMON --method ccs --seed $seed --no_stab

    echo "=== Amazon ICS no_collab seed=$seed ==="
    $PY $RUN $COMMON --method ccs --seed $seed --no_collab
done

echo "=== Amazon CCS ablation v2 done ==="
