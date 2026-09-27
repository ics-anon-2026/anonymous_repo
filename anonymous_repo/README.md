# Invariant Collaborative Sparsification (ICS) — Reproducibility Guide

Anonymous code repository for the paper *Invariant Collaborative Sparsification for Robust Fraud Detection on Heterogeneous Graphs*.

## Environment

- Python 3.8+ with PyTorch >= 1.12, PyTorch Geometric >= 2.0, scikit-learn, scipy, pandas
- CPU is sufficient for all experiments in the paper (no GPU required)
- Tested on: Windows 10, Python 3.13 (Miniconda), PyTorch 2.x, PyG 2.x

```bash
pip install torch torch_geometric scikit-learn scipy pandas
```

## Data

Download the datasets and place them under `data/`:

1. **YelpChi & Amazon**: from the [CARE-GNN repository](https://github.com/YingtongDou/CARE-GNN) (`data/YelpChi.mat`, `data/Amazon.mat`), or the [PC-GNN repository](https://github.com/P FraudDetection/PC-GNN). Place under `data/raw/Fraud/`.
2. **Elliptic Bitcoin**: from [Kaggle](https://www.kaggle.com/ellipticco/elliptic-data-set). Place `elliptic_bitcoin_dataset/` under `data/elliptic/`.

The raw `.mat` files are converted to PyTorch-Geometric `Data` objects on first run and cached under `data/processed/`.

## Repository Layout

```
src/direction2/
  models/            # BinaryGAT, GCN, GAT, HAN, PC-GNN sampler, CausalCollaborativeSparsification
  data/              # FraudDataset, EllipticDataset, ood_split
  utils/             # train_eval (BCE + balanced batches + gradient clipping), graph_utils
  experiments/
    run_ccs.py       # MAIN ENTRY: all IID main-table, ablation, sensitivity runs
    run_attack.py    # structural attack experiments (DICE / Nettack-style)
    run_significance.py  # paired t-test / Wilcoxon over the 5-seed results
papers/direction2/   # paper source (paper.tex), figures, generate_docx.py
results/direction2/  # all experiment CSVs (one file per run)
```

## Reproducing the Main Results (Table 1)

All runs use a fixed random 50%/25%/25% split and report mean ± std over 5 seeds (0–4).

```bash
PY=python   # your python
RUN=src/direction2/experiments/run_ccs.py
COMMON="--model BinaryGAT --balanced_batch --use_val_threshold --epochs 100 --lr 0.001 --dropout 0.1"

# YelpChi / Amazon: Random, ICS, PC-GNN  (seeds 0-4)
for DS in YelpChi Amazon; do
  for SEED in 0 1 2 3 4; do
    python $RUN --dataset $DS --method random  --topk 20 --seed $SEED $COMMON
    python $RUN --dataset $DS --method ccs    --topk 20 --n_envs 3 --seed $SEED $COMMON
    python $RUN --dataset $DS --method pcgnn  --topk 20 --pcgnn_pos_ratio 0.5 --seed $SEED $COMMON
  done
done

# Elliptic: Random / ICS (topk=10, seeds 0-2) and full graph (topk=100)
for SEED in 0 1 2; do
  python $RUN --dataset Elliptic --method random --topk 10 --seed $SEED $COMMON
  python $RUN --dataset Elliptic --method ccs    --topk 10 --n_envs 3 --seed $SEED $COMMON
done
python $RUN --dataset Elliptic --method full --topk 100 --seed 0 $COMMON
```

## Reproducing the Ablation (Table 2)

```bash
for DS in YelpChi Amazon; do
  for SEED in 0 1 2 3 4; do
    python $RUN --dataset $DS --method ccs --topk 20 --n_envs 3 --seed $SEED --no_stab --no_collab $COMMON   # sim only
    python $RUN --dataset $DS --method ccs --topk 20 --n_envs 3 --seed $SEED --no_stab $COMMON                # sim+stab
    python $RUN --dataset $DS --method ccs --topk 20 --n_envs 3 --seed $SEED --no_collab $COMMON              # sim+collab
  done
done
```

## Reproducing OOD Generalization (Table 3)

```bash
for DS in YelpChi Amazon; do
  for SEED in 0 1 2; do
    python $RUN --dataset $DS --method ccs    --topk 20 --n_envs 3 --seed $SEED --ood_split $COMMON
    python $RUN --dataset $DS --method random --topk 20 --seed $SEED --ood_split $COMMON
  done
done
```

## Reproducing Structural Attack Robustness (Table 4)

```bash
python src/direction2/experiments/run_attack.py --all --budget 0.05   # 40 runs: 2 datasets x 2 attacks x 2 defenses x 5 seeds... (seeds 0-2 via --all; add 3-4 with run_attack_seeds34.py)
```

## Reproducing Parameter Sensitivity (Figure 9–10)

```bash
python src/direction2/experiments/run_lambda_beta_grid.py      # seed 0
python src/direction2/experiments/run_lambda_beta_seeds.py     # seeds 1-4
python src/direction2/experiments/combine_lambda_beta.py       # merge + heatmap
# k / M sensitivity: vary --topk {10,20,50} and --n_envs {2,3,5,10} with seed 0
```

## Significance Tests

```bash
python src/direction2/experiments/run_significance.py
```

## Notes

- The sparsification preprocessing (`CausalCollaborativeSparsification`) is deterministic given the data: K-means uses `random_state=42`, so environments are a fixed property of each dataset; only model initialization varies across seeds.
- All reported numbers in the paper are mean ± std (population std, `np.std`) over the seeds listed above.
- A full re-run of every experiment in the paper takes approximately 3–4 hours on a modern CPU.
