# UDA Cross-Subject EEG-MI Framework

This folder contains a self-contained implementation of the experimental
framework from the paper:

> *Unsupervised Domain Adaptation for Cross-Subject Generalization in EEG-Based
> Brain-Computer Interfaces: A Multi-Dataset Study*

The existing pipeline (denoising, dataset loader, dashboard, etc.) has **not**
been modified. All outputs go to a separate `results_uda/` directory.

## What is implemented

| Paper section | Implementation |
|---|---|
| 4.2 Source / target domains | `data.py` → `DomainFold`, `make_loso_folds()` |
| 4.3 Preprocessing (8–30 Hz, notch, CAR) | `data.py` → `preprocess_trials()` |
| 4.4 Source-only baseline | `trainer.py` → `source_only` loss |
| 4.5 CORAL `L = L_cls + λ_coral·L_CORAL` | `losses.py` → `coral_loss()` |
| 4.5 MMD `L = L_cls + λ_mmd·L_MMD` | `losses.py` → `mmd_loss()` |
| 4.6 DANN (F + C + D, gradient reversal) | `models.py` → `UDANet`, `losses.py` → `GradientReversalLayer` |
| 4.7 LOSO with separate adaptation/test partitions | `data.py`, `evaluation.py` |
| 4.8 Classification metrics + domain alignment | `metrics.py` |
| 4.9 Target data availability (25/50/75/100%) | `data.py` → `select_target_proportion()` |
| 5.5 / 6.3 Paired statistics (Wilcoxon + Holm) | `evaluation.py` → `paired_statistics()` |

## Files

```
uda_cross_subject/
├── config.py        # all hyperparameters (paper's "to be determined" values)
├── data.py          # MOABB/synthetic loading, preprocessing, LOSO folds
├── features.py      # FBCSP (fit on source labels) + scaling (source + unlabelled target)
├── losses.py        # CORAL, multi-kernel MMD, gradient reversal
├── models.py        # F (feature extractor), C (classifier), D (domain discriminator)
├── trainer.py       # four training loops, GRL schedule, source-val model selection
├── metrics.py       # accuracy/balanced accuracy/macro P-R-F1/kappa + alignment
├── evaluation.py    # full experiment, CSV tables, aggregation + statistics
├── reporting.py     # timestamped run folder, Excel workbooks, fig1-fig6 PNGs
├── run_uda.py       # command-line interface
└── README.md
```

## How to run

**1. Quick smoke test (synthetic data, no download required):**
```
py -m uda_cross_subject.run_uda --mode synthetic
```

**2. Real dataset (MOABB), all four methods:**
```
py -m uda_cross_subject.run_uda --mode dataset --dataset bci2a --subjects 1 2 3
```

**3. Full target-data availability analysis from the paper:**
```
py -m uda_cross_subject.run_uda --mode dataset --dataset bci2a \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --proportions 0.25 0.5 0.75 1.0 --epochs 80
```

**4. Run every supported dataset in one command:**
```
py -m uda_cross_subject.run_uda --mode dataset --dataset all \
    --subjects 1 2 3 --proportions 0.25 0.5 0.75 1.0 --epochs 80
```

`--dataset` accepts several names too, e.g.
`--dataset bci2a physionet openbmi`. The supported keys (13) come directly from
`data_pipeline/dataset_fetcher.py`: `physionet`, `bci2a`, `openbmi`, `weibo2014`,
`bci2b`, `bci4`, `ofner2017`, `zhou2016`, `cho2017`, `shin2017`, `alexmi`,
`grossewentrup2009`, `sosulski2019`.

Each dataset gets its own timestamped `run_*` folder. One combined
`results_uda_summary_<timestamp>.xlsx` is written at the end for all of them.
If one dataset fails to load or has incompatible subject IDs, it is skipped and
the remaining datasets continue; a final summary lists completed and failed
datasets. Keep `--subjects` conservative (e.g. `1 2 3`) because not every
dataset contains high subject IDs, and the first run of each dataset downloads
and caches its data via MOABB.

The module can also be run directly as a script:
```
py uda_cross_subject/run_uda.py --mode synthetic
```

## Outputs — separate folder, the existing `results/` is untouched

Each execution creates its own **timestamped run folder**, following the same
convention as the original pipeline:

```
results_uda/
├── run_20261122_141459_synthetic/
│   ├── subject_level.csv
│   ├── summary.csv
│   ├── alignment.csv
│   ├── statistics.csv
│   ├── config.json
│   ├── synthetic_uda_results_<timestamp>.xlsx
│   ├── fig1_method_comparison.png
│   ├── fig2_target_data_effect.png
│   ├── fig3_domain_alignment.png
│   ├── fig4_dann_discriminator.png
│   ├── fig5_subject_variability.png
│   └── fig6_confusion_matrices.png
└── results_uda_summary_<timestamp>.xlsx   # combined workbook for all runs
```

| File | Contents |
|---|---|
| `subject_level.csv` | Metrics and confusion counts per target subject × method × target proportion |
| `summary.csv` | Mean ± std of every method across target subjects |
| `alignment.csv` | Covariance discrepancy / MMD (before → after), DANN domain accuracy |
| `statistics.csv` | Each UDA method vs source-only: mean difference, Wilcoxon p, Holm-adjusted p |
| `config.json` | All run parameters (for reproducibility) |
| `*_uda_results_*.xlsx` | Sheets: Overview, Summary, Subject Results, Alignment, Statistics, Config |
| `results_uda_summary_*.xlsx` | Summary of all run folders in one workbook |
| `fig1`–`fig6` | Accuracy, target-data effect, alignment, DANN discriminator, subject variability, confusion matrices |

The figures use the same styling as the existing `evaluation/visualization.py`
(serif font, 300 dpi, `fig*.png` naming). Use `--output-dir` to change the
output root.

## Dashboard integration

The existing web dashboard can browse and launch UDA runs:

```
python dashboard/backend.py
# open http://localhost:5000 and select the "UDA Results" tab
```

It lists every `results_uda/run_*` folder, plots method comparison, target-data
availability and alignment metrics, shows the paired statistics tables and the
six generated figures, and can start a new run (including `--dataset all`) from
the sidebar. Backend endpoints: `/api/uda/runs`, `/api/uda/run/<name>`,
`/api/uda/figure/<run>/<file>`, `POST /api/uda/run`, `POST /api/uda/stop`.

## Important notes

- **Target test labels** are used only for final evaluation after training —
  never during adaptation, scaling, model selection or hyperparameter tuning.
- FBCSP is fitted on **source labels only**. The scaler is fitted on source
  training features plus the unlabelled target adaptation features. The test
  partition is never used for fitting.
- All methods share the same feature set and initial weights (fair comparison).
- DANN uses the standard gradient-reversal schedule
  `λ = 2/(1+exp(−10p)) − 1`.
- `--target-split chronological` builds the test partition in recording order
  to reduce temporal leakage. MOABB metadata does not expose session IDs, so a
  fully session-aware split is not possible.
- Synthetic mode is **binary** and intended only for framework validation;
  paper results should be produced from real datasets.
- The paper leaves hyperparameters as "to be determined" — reasonable defaults
  are provided here and can be changed with `--lambda-*`, `--epochs`,
  `--feature-dim`, etc.
- **CORAL λ defaults to 1000** because features are standardised and the paper's
  `1/(4d²)` factor makes the raw loss very small (~1e-5); with λ=1, adaptation
  has virtually no effect. Tune it for your own features.
