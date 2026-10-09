# ICS — Training-Free Graph Sparsification as a Robust Defense against Structural Attacks on Fraud-Detection Graphs

Code and evaluation protocol for the systematic benchmark described in the manuscript
submitted to *Discover Artificial Intelligence* (Springer). It accompanies the paper
"Training-Free Graph Sparsification as a Robust Defense against Structural Attacks on
Fraud Detection Graphs: A Systematic Benchmark".

## Method

**ICS (Invariant Collaborative Sparsification)** is a training-free, label-free edge-selection
method for fraud-detection graphs. Every edge is scored by (i) feature consistency,
(ii) cross-environment stability, and (iii) an optional multi-relation collaborative bonus;
the top-`k` edges are kept *before* any downstream model is trained. Because it consumes only
features and structure, it can be composed with any graph model (here: BinaryGAT).

## Repository layout

- `src/direction2/` — ICS implementation, baselines (random top-`k`, KCES, GraphConsis,
  a PC-GNN-style sampler), datasets (YelpChi, Amazon, Elliptic), training/eval, and the
  attack benchmark entry points:
  - `experiments/run_attack.py` — run one (dataset, attack, defense, seed) combination.
  - `experiments/phase2_launcher.py` — run the full 198-combination matrix; spawns one
    isolated subprocess per combination, honors a wall-clock budget (`--max-hours`), and
    resumes from already-produced CSVs.
  - `experiments/aggregate_attack_results.py` — aggregate all produced CSVs into the
    summary table (AUC mean ± std over seeds).
- `src/common/attacks/` — unified attack dispatcher `connect.py` and the implementations of
  CAMOUFLAGE, PRBCD, Metattack, and a BinarizedAttack-style structural perturbation, plus
  `resource_limiter.py` (CPU/memory guard). `connect.py` imports `deeprobust_wrapper` from
  this same folder.
- `paper/paper_discover.tex` — LaTeX source of the submitted manuscript.
- `RESULTS.md` — the aggregated 198-run attack benchmark (AUC mean ± std across seeds).

## Dependencies

- Python 3.12+
- `torch`, `torch_geometric`
- `deeprobust` (provides PRBCD / Metattack / DICE / Nettack)
- `scikit-learn`, `numpy`, `scipy`, `matplotlib`

The scripts add `src/direction2/{data,models,utils}` and `src/common/attacks` to `sys.path`
automatically, so no manual `PYTHONPATH` setup is required.

## Datasets

- **YelpChi** and **Amazon**: public from the CARE-GNN repository.
- **Elliptic**: public from <https://www.kaggle.com/ellipticco/elliptic-data-set>.

Place the raw data under `src/direction2/data/` (see `download_datasets.py`).

## Reproduce

Single combination:

```bash
cd src/direction2/experiments
python run_attack.py --dataset YelpChi --attack camo --defense ics --seed 0
```

Full 198-run matrix (hard-coded across Amazon / YelpChi / Elliptic, 5 structural attacks,
4 sparsifiers, 5 seeds; resumes from existing outputs):

```bash
cd src/direction2/experiments
python phase2_launcher.py --max-hours 8
```

Aggregate the produced CSVs:

```bash
cd src/direction2/experiments
python aggregate_attack_results.py
```

## Notes

- Sparsification is deterministic given the data: K-means uses a fixed random state, so
  environments are a fixed property of each dataset; only model initialization varies across seeds.
- All reported numbers are mean ± std (population std) over the seeds above.
- A full CPU re-run of the 198-combination matrix is segmented into wall-clock-bounded
  batches via `--max-hours`; each combination runs in an isolated subprocess so memory is
  reclaimed cleanly between runs.

## License

Released for the reproducibility of the submitted work.
