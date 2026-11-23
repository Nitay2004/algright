"""Full LOSO experiment: four methods, alignment analysis, target-data effect.

Implements paper sections 4.7-4.9, 5.4, 5.5, 6.1-6.4.
Existing project files are untouched.
"""

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix

from . import reporting
from .config import UDAConfig
from .data import make_loso_folds, select_target_proportion
from .features import FeatureBuilder
from .metrics import (
    classification_metrics,
    domain_discriminator_accuracy,
    representation_metrics,
)
from .models import build_model
from .trainer import predict, resolve_device, UDATrainer

METRIC_COLUMNS = [
    "accuracy", "balanced_accuracy", "precision_macro",
    "recall_macro", "f1_macro", "cohen_kappa",
]
ALIGN_COLUMNS = ["covariance_discrepancy", "mmd"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _representations(model, features, cfg: UDAConfig):
    device = next(model.parameters()).device
    model.eval()
    zs = model.extract(
        torch.as_tensor(np.asarray(features.source_train, dtype=np.float32),
                        device=device)
    ).cpu().numpy()
    zt = model.extract(
        torch.as_tensor(np.asarray(features.target_adapt, dtype=np.float32),
                        device=device)
    ).cpu().numpy()
    return zs, zt


def _alignment_values(model, features, cfg: UDAConfig, method: str):
    zs, zt = _representations(model, features, cfg)
    values = representation_metrics(zs, zt, cfg)
    domain_acc = float("nan")
    if method == "dann":
        domain_acc = domain_discriminator_accuracy(model, features, cfg)
    return values, domain_acc


def evaluate_fold(fold, features, cfg: UDAConfig, dataset_name: str,
                  proportion: float):
    """Train/evaluate every method on one fold. Returns two row lists."""
    result_rows, alignment_rows = [], []
    n_classes = int(len(np.unique(features.source_train_labels)))
    input_dim = features.source_train.shape[1]

    for method in cfg.methods:
        # Identical initial weights across methods => fair comparison.
        set_seed(cfg.seed)
        model = build_model(input_dim, n_classes, cfg, method).to(
            resolve_device(cfg.device)
        )

        before, domain_before = _alignment_values(model, features, cfg, method)
        trainer = UDATrainer(model, cfg, method)
        history = trainer.fit(features)
        after, domain_after = _alignment_values(model, features, cfg, method)

        proba, pred = predict(model, features.target_test, cfg)
        cls = classification_metrics(
            fold.target_test_labels, pred, n_classes=n_classes
        )

        row = {
            "dataset": dataset_name,
            "target_subject": fold.target_subject,
            "method": method,
            "target_proportion": proportion,
            "n_source": fold.n_source,
            "n_target_adapt": fold.n_target_adapt,
            "n_target_test": len(fold.target_test_labels),
            "feature_kind": features.kind,
            **cls,
            "final_train_loss": history["train_loss"][-1],
            "best_val_loss": history["best_val_loss"],
            "epochs_run": history["epochs_run"],
        }
        # Aggregated confusion-matrix counts (used by fig6_confusion_matrices).
        cm = confusion_matrix(
            fold.target_test_labels, pred, labels=np.arange(n_classes)
        )
        for i in range(n_classes):
            for j in range(n_classes):
                row[f"cm_{i}{j}"] = int(cm[i, j])
        result_rows.append(row)

        alignment_rows.append({
            "dataset": dataset_name,
            "target_subject": fold.target_subject,
            "method": method,
            "target_proportion": proportion,
            "covariance_discrepancy_before": before["covariance_discrepancy"],
            "covariance_discrepancy_after": after["covariance_discrepancy"],
            "mmd_before": before["mmd"],
            "mmd_after": after["mmd"],
            "domain_acc_before": domain_before,
            "domain_acc_after": domain_after,
        })

        if cfg.verbose:
            print(f"    {method:<11s} acc={cls['accuracy']:.4f} "
                  f"f1={cls['f1_macro']:.4f} kappa={cls['cohen_kappa']:.4f} "
                  f"| cov {before['covariance_discrepancy']:.4f} -> "
                  f"{after['covariance_discrepancy']:.4f}")
    return result_rows, alignment_rows


def aggregate_summary(subject_df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across target subjects for each dataset/method/proportion."""
    keys = ["dataset", "target_proportion", "method"]
    agg = {f"{m}_mean": (m, "mean") for m in METRIC_COLUMNS}
    agg.update({f"{m}_std": (m, "std") for m in METRIC_COLUMNS})
    agg["n_subjects"] = ("target_subject", "nunique")
    return subject_df.groupby(keys, as_index=False).agg(**agg).sort_values(keys)


def _holm_adjust(pvalues):
    """Holm-Bonferroni step-down correction; NaNs are preserved."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    finite = np.where(np.isfinite(p))[0]
    if len(finite) == 0:
        return out
    order = finite[np.argsort(p[finite])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        out[idx] = min(1.0, running)
    return out


def paired_statistics(subject_df: pd.DataFrame, cfg: UDAConfig,
                      metric: str = "accuracy") -> pd.DataFrame:
    """Paired subject-level UDA vs source-only tests (paper 4.9, 6.3)."""
    rows = []
    for (dataset, proportion), group in subject_df.groupby(
            ["dataset", "target_proportion"]):
        pivot = group.pivot_table(
            index="target_subject", columns="method", values=metric
        )
        if "source_only" not in pivot.columns:
            continue
        for method in cfg.methods:
            if method == "source_only" or method not in pivot.columns:
                continue
            paired = pivot[[method, "source_only"]].dropna()
            diff = (paired[method] - paired["source_only"]).values
            mean_diff = float(np.mean(diff)) if len(diff) else float("nan")
            p_value = float("nan")
            if len(diff) >= 2 and np.any(np.abs(diff) > 1e-12):
                try:
                    p_value = float(wilcoxon(diff, zero_method="wilcox").pvalue)
                except ValueError:
                    p_value = 1.0
            elif len(diff) >= 2:
                p_value = 1.0
            rows.append({
                "dataset": dataset,
                "target_proportion": proportion,
                "method": method,
                "metric": metric,
                "n_pairs": int(len(diff)),
                "mean_difference": mean_diff,
                "p_value": p_value,
            })

    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats["p_value_holm"] = _holm_adjust(stats["p_value"].values)
        stats["significant_0.05"] = stats["p_value_holm"] < 0.05
    return stats


def print_summary(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("  CROSS-SUBJECT UDA SUMMARY (mean +/- std over target subjects)")
    print("=" * 88)
    header = (f"{'Dataset':<12} {'Target %':>8} {'Method':<12} "
              f"{'Accuracy':>16} {'F1':>16} {'Kappa':>16}")
    print(header)
    print("-" * 88)
    for _, row in summary.iterrows():
        print(f"{row['dataset']:<12} {row['target_proportion']:>8.0%} "
              f"{row['method']:<12} "
              f"{row['accuracy_mean']:>7.4f} +/-{row['accuracy_std']:<7.4f} "
              f"{row['f1_macro_mean']:>7.4f} +/-{row['f1_macro_std']:<7.4f} "
              f"{row['cohen_kappa_mean']:>7.4f} +/-{row['cohen_kappa_std']:<7.4f}")
    print("=" * 88 + "\n")


def run_experiment(X, y, meta, cfg: UDAConfig, write_combined: bool = True):
    """Run the complete LOSO experiment and persist all result tables.

    ``write_combined=False`` is used by multi-dataset runs so that the root
    summary workbook is written once, after every dataset has finished.
    """
    dataset_name = str(meta.get("dataset_name", cfg.dataset_name))
    folds = list(make_loso_folds(X, y, meta, cfg))
    if not folds:
        raise RuntimeError("No eligible LOSO folds were constructed.")

    subject_rows, alignment_rows = [], []
    for fold in folds:
        print(f"[UDA] Target subject {fold.target_subject} "
              f"| source={fold.n_source} trials "
              f"| target_adapt={fold.n_target_adapt} "
              f"| target_test={len(fold.target_test_labels)}")

        for proportion in cfg.target_proportions:
            fold_p = select_target_proportion(fold, proportion, cfg)
            print(f"  Target adaptation data: {proportion:.0%} "
                  f"({fold_p.n_target_adapt} unlabelled trials)")
            features = FeatureBuilder(cfg, meta["sfreq"]).fit_transform(fold_p)
            rows, align = evaluate_fold(
                fold_p, features, cfg, dataset_name, proportion
            )
            subject_rows.extend(rows)
            alignment_rows.extend(align)

    subject_df = pd.DataFrame(subject_rows)
    alignment_df = pd.DataFrame(alignment_rows)
    summary = aggregate_summary(subject_df)
    stats = paired_statistics(subject_df, cfg, metric="accuracy")

    base_dir = Path(cfg.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = reporting.make_run_dir(base_dir, cfg.mode, dataset_name)
    tables = {
        "subject_level": subject_df,
        "alignment": alignment_df,
        "summary": summary,
        "statistics": stats,
    }

    subject_df.to_csv(run_dir / "subject_level.csv", index=False)
    alignment_df.to_csv(run_dir / "alignment.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    stats.to_csv(run_dir / "statistics.csv", index=False)
    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)

    stamp = reporting.run_stamp(run_dir)
    workbook = reporting.write_workbook(
        run_dir / f"{dataset_name}_uda_results_{stamp}.xlsx",
        cfg, tables, dataset_name, run_dir,
    )
    figures = reporting.generate_all_figures(tables, run_dir)
    combined = (reporting.write_combined_workbook(base_dir, tables, stamp=stamp)
                if write_combined else None)

    print_summary(summary)
    print(f"[UDA] Results saved to: {run_dir.resolve()}")
    print(f"[UDA] Excel: {workbook}")
    if combined:
        print(f"[UDA] Combined Excel: {combined}")
    print(f"[UDA] Figures: {len(figures)} PNG file(s)")
    return {
        **tables,
        "run_dir": run_dir,
        "workbook": workbook,
        "combined_workbook": combined,
        "figures": figures,
    }
