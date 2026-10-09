#!/bin/bash
# Multi-seed experiment runner for ICS paper revision
# Usage: bash run_multi_seed.sh

PYTHON="D:/workbuddy工作区/SCI论文发表/.venv/Scripts/python.exe"
SCRIPT="D:/workbuddy工作区/SCI论文发表/src/direction2/experiments/run_ccs.py"
N_SEEDS=5
EPOCHS=40

echo "=== ICS Paper Revision Experiments ==="
echo "Datasets: YelpChi, Amazon  (Elliptic to be added)"
echo "Models:   GAT, GCN"
echo "Methods:  ics, random, dropedge"
echo "Seeds:    $N_SEEDS"
echo ""

mkdir -p results_dir2

# ---- Main Results: GAT + ICS vs Random vs DropEdge on YelpChi ----
echo "--- YelpChi GAT ---"
for seed in $(seq 0 $((N_SEEDS-1))); do
    for method in random dropedge; do
        echo "  seed=$seed method=$method"
        $PYTHON "$SCRIPT" --dataset YelpChi --model GAT --method $method --topk 20 --seed $seed --epochs $EPOCHS --lr 5e-4
    done
    echo "  seed=$seed method=ccs"
    $PYTHON "$SCRIPT" --dataset YelpChi --model GAT --method ccs --topk 20 --seed $seed --epochs $EPOCHS --lr 5e-4 --collab_bonus 0.1
    # Ablations
    $PYTHON "$SCRIPT" --dataset YelpChi --model GAT --method ccs --topk 20 --seed $seed --epochs $EPOCHS --lr 5e-4 --no_stab
    $PYTHON "$SCRIPT" --dataset YelpChi --model GAT --method ccs --topk 20 --seed $seed --epochs $EPOCHS --lr 5e-4 --no_collab
    $PYTHON "$SCRIPT" --dataset YelpChi --model GAT --method ccs --topk 20 --seed $seed --epochs $EPOCHS --lr 5e-4 --no_stab --no_collab
done

# ---- Main Results: GAT + ICS vs Random on Amazon ----
echo "--- Amazon GAT ---"
for seed in $(seq 0 $((N_SEEDS-1))); do
    echo "  seed=$seed method=random"
    $PYTHON "$SCRIPT" --dataset Amazon --model GAT --method random --topk 20 --seed $seed --epochs $EPOCHS --lr 5e-4
    echo "  seed=$seed method=ccs"
    $PYTHON "$SCRIPT" --dataset Amazon --model GAT --method ccs --topk 20 --seed $seed --epochs $EPOCHS --lr 5e-4 --collab_bonus 0.1
done

# ---- Parameter Sensitivity: vary topk ----
echo "--- Sensitivity: topk ---"
for k in 5 10 20 50 100; do
    echo "  YelpChi GAT ccs topk=$k"
    $PYTHON "$SCRIPT" --dataset YelpChi --model GAT --method ccs --topk $k --seed 42 --epochs $EPOCHS --lr 5e-4 --collab_bonus 0.1
done

# ---- Parameter Sensitivity: vary n_envs ----
echo "--- Sensitivity: n_envs ---"
for m in 2 5 8 10; do
    echo "  YelpChi GAT ccs n_envs=$m"
    $PYTHON "$SCRIPT" --dataset YelpChi --model GAT --method ccs --topk 20 --n_envs $m --seed 42 --epochs $EPOCHS --lr 5e-4 --collab_bonus 0.1
done

# ---- Parameter Sensitivity: vary lambda (stability weight) ----
echo "--- Sensitivity: lam ---"
# Need to add --lam support to run_ccs.py
# For now, use ablation approach

echo ""
echo "=== All experiments complete ==="
echo "Results saved in results/direction2/"
