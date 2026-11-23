"""Output reporting for the UDA framework.

Mirrors the original pipeline's result convention:

* every execution gets its own timestamped ``run_*`` folder,
* CSV tables + JSON config + an rich Excel workbook live inside that folder,
* publication-style ``fig*.png`` figures are written next to them,
* a combined ``results_uda_summary_<timestamp>.xlsx`` is written at the root
  of the UDA results directory (never inside the original ``results/`` folder).

Nothing here writes to the existing project's ``results/`` directory.
"""

import datetime
import os
from copy import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from openpyxl.styles import Alignment as XlAlignment, Font  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "source_only": "#2c3e50",
    "coral": "#3498db",
    "mmd": "#f39c12",
    "dann": "#9b59b6",
}
METHOD_ORDER = ["source_only", "coral", "mmd", "dann"]


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
def make_run_dir(base_dir, mode: str, dataset_name: str) -> Path:
    """Create (and return) a fresh run_* folder under the UDA results root."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = str(mode) if str(mode) == str(dataset_name) else f"{mode}_{dataset_name}"
    folder = f"run_{stamp}_{tag}".replace(" ", "_")
    run_dir = Path(base_dir) / folder
    suffix = 1
    while run_dir.exists():  # two runs within the same second
        run_dir = Path(base_dir) / f"{folder}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_stamp(run_dir: Path) -> str:
    return run_dir.name.replace("run_", "", 1)


# ----------------------------------------------------------------------
# Excel
# ----------------------------------------------------------------------
def _style_header(ws):
    for cell in ws[1]:
        f = copy(cell.font)
        f.bold = True
        cell.font = f
        cell.alignment = XlAlignment(horizontal="center", vertical="center")
    for col_cells in ws.columns:
        letter = col_cells[0].column_letter
        width = max(len(str(c.value)) if c.value is not None else 0
                    for c in col_cells)
        ws.column_dimensions[letter].width = min(max(width + 2, 10), 42)


def _round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(6)
        if out[col].dtype == bool:
            out[col] = out[col].map({True: "yes", False: "no"})
    return out


def _safe_sheet(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")[:31]


def _config_rows(cfg) -> pd.DataFrame:
    from dataclasses import asdict
    rows = [(k, ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v)
            for k, v in asdict(cfg).items()]
    return pd.DataFrame(rows, columns=["Config", "Value"])


def _overview_rows(cfg, tables, dataset_name, run_dir):
    summary = tables["summary"]
    valid = summary.dropna(subset=["accuracy_mean"])
    best_row = valid.loc[valid["accuracy_mean"].idxmax()] if not valid.empty else None
    rows = [
        ("Dataset", dataset_name),
        ("Mode", cfg.mode),
        ("Run folder", str(run_dir)),
        ("Methods", ", ".join(cfg.methods)),
        ("Target proportions", ", ".join(f"{p:.0%}" for p in cfg.target_proportions)),
        ("Target subjects", int(summary["n_subjects"].max()) if "n_subjects" in summary else 0),
        ("Best method (accuracy)", best_row["method"] if best_row is not None else ""),
        ("Best accuracy", f"{best_row['accuracy_mean']:.4f}" if best_row is not None else ""),
        ("Generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def write_workbook(out_path, cfg, tables, dataset_name, run_dir):
    """One workbook per run: Overview / Summary / Subject Results / ... ."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        sheets = [
            ("Overview", _overview_rows(cfg, tables, dataset_name, run_dir)),
            ("Summary", tables["summary"]),
            ("Subject Results", tables["subject_level"]),
            ("Alignment", tables["alignment"]),
            ("Statistics", tables["statistics"]),
            ("Config", _config_rows(cfg)),
        ]
        for name, frame in sheets:
            frame = _round_numeric(frame) if isinstance(frame, pd.DataFrame) else frame
            sheet_name = _safe_sheet(name)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            _style_header(writer.sheets[sheet_name])
    return out_path


def write_combined_workbook(base_dir, tables=None, stamp=None):
    """Aggregate every run_* summary.csv under base_dir into one workbook.

    ``tables`` is optional; the workbook is built from the saved CSV files so
    that multi-dataset runs can call this once after all datasets finish.
    """
    base = Path(base_dir)
    frames = []
    for csv in sorted(base.glob("run_*/summary.csv")):
        try:
            df = pd.read_csv(csv)
            df["run"] = csv.parent.name
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    stamp = stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"results_uda_summary_{stamp}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        _round_numeric(combined).to_excel(writer, sheet_name="Summary", index=False)
        _style_header(writer.sheets["Summary"])
    return out


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _save(fig, out_dir: Path, filename: str):
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _order_methods(values):
    return [m for m in METHOD_ORDER if m in values] + [
        m for m in values if m not in METHOD_ORDER
    ]


def fig1_method_comparison(summary, out_dir):
    proportions = sorted(summary["target_proportion"].unique())
    n = max(len(proportions), 1)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    fig.suptitle("Figure 1: Cross-Subject Classification Accuracy (mean +/- std)",
                 fontweight="bold")

    for ax, prop in zip(axes[0], proportions):
        sub = summary[summary["target_proportion"] == prop]
        methods = _order_methods(sub["method"].unique())
        means = [sub.loc[sub["method"] == m, "accuracy_mean"].mean()
                 for m in methods]
        stds = [sub.loc[sub["method"] == m, "accuracy_std"].mean()
                for m in methods]
        colors = [COLORS.get(m, "#7f8c8d") for m in methods]
        bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=4,
                      color=colors, edgecolor="#2c3e50", linewidth=0.6, alpha=0.9)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, min(1.05, max(means + [0.1]) + 0.12))
        ax.set_title(f"Target adaptation data = {prop:.0%}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, out_dir, "fig1_method_comparison.png")


def fig2_target_data_effect(summary, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Figure 2: Effect of Unlabelled Target Data Availability",
                 fontweight="bold")
    for method in _order_methods(summary["method"].unique()):
        sub = summary[summary["method"] == method].sort_values("target_proportion")
        if sub.empty:
            continue
        ax.errorbar(sub["target_proportion"], sub["accuracy_mean"],
                    yerr=sub["accuracy_std"], marker="o", linewidth=1.8,
                    capsize=4, color=COLORS.get(method, "#7f8c8d"), label=method)
    ax.set_xlabel("Fraction of unlabelled target adaptation data")
    ax.set_ylabel("Accuracy (mean +/- std)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save(fig, out_dir, "fig2_target_data_effect.png")


def fig3_domain_alignment(alignment, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle("Figure 3: Source-Target Domain Alignment Before vs After Adaptation",
                 fontweight="bold")
    methods = _order_methods(alignment["method"].unique())
    x = np.arange(len(methods))
    width = 0.35

    for ax, key, label in zip(
            axes,
            ["covariance_discrepancy", "mmd"],
            ["Covariance discrepancy", "Kernel MMD"]):
        before = [alignment.loc[alignment["method"] == m, f"{key}_before"].mean()
                  for m in methods]
        after = [alignment.loc[alignment["method"] == m, f"{key}_after"].mean()
                 for m in methods]
        ax.bar(x - width / 2, before, width, label="Before",
               color="#bdc3c7", edgecolor="#2c3e50", linewidth=0.5)
        ax.bar(x + width / 2, after, width, label="After",
               color="#3498db", edgecolor="#2c3e50", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, out_dir, "fig3_domain_alignment.png")


def fig4_dann_discriminator(alignment, out_dir):
    sub = alignment[alignment["method"] == "dann"]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle("Figure 4: DANN Domain-Discriminator Accuracy", fontweight="bold")
    props = sorted(sub["target_proportion"].unique())
    x = np.arange(len(props))
    width = 0.35
    before = [sub.loc[sub["target_proportion"] == p, "domain_acc_before"].mean()
              for p in props]
    after = [sub.loc[sub["target_proportion"] == p, "domain_acc_after"].mean()
             for p in props]
    ax.bar(x - width / 2, before, width, label="Before adaptation",
           color="#bdc3c7", edgecolor="#2c3e50", linewidth=0.5)
    ax.bar(x + width / 2, after, width, label="After adaptation",
           color="#9b59b6", edgecolor="#2c3e50", linewidth=0.5)
    for i, value in enumerate(after):
        if np.isfinite(value):
            ax.text(i + width / 2, value + 0.01, f"{value:.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0%}" for p in props])
    ax.set_xlabel("Target adaptation data")
    ax.set_ylabel("Domain accuracy (lower = better alignment)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, out_dir, "fig4_dann_discriminator.png")


def fig5_subject_variability(subject_df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.suptitle("Figure 5: Subject-Level Accuracy Variability", fontweight="bold")
    methods = _order_methods(subject_df["method"].unique())
    data = [subject_df.loc[subject_df["method"] == m, "accuracy"].dropna().values
            for m in methods]
    colors = [COLORS.get(m, "#7f8c8d") for m in methods]
    box = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Target-subject accuracy")
    ax.set_ylim(0, 1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, out_dir, "fig5_subject_variability.png")


def fig6_confusion_matrices(subject_df, out_dir):
    """Aggregated target-test confusion matrices, one panel per method."""
    cm_cols = [c for c in subject_df.columns if c.startswith("cm_")]
    if not cm_cols:
        return None
    n_classes = int(np.sqrt(len(cm_cols)))
    if n_classes * n_classes != len(cm_cols):
        return None
    methods = _order_methods(subject_df["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4.2),
                             squeeze=False)
    fig.suptitle("Figure 6: Aggregated Target-Test Confusion Matrices",
                 fontweight="bold")
    for ax, method in zip(axes[0], methods):
        sub = subject_df[subject_df["method"] == method]
        cm = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                col = f"cm_{i}{j}"
                if col in sub.columns:
                    cm[i, j] = float(sub[col].sum())
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(method)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        threshold = cm.max() / 2.0 if cm.max() > 0 else 0
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        color="white" if cm[i, j] > threshold else "black",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return _save(fig, out_dir, "fig6_confusion_matrices.png")


def generate_all_figures(tables, out_dir):
    print("[UDA] Generating figures ...")
    paths = []
    for fn in (fig1_method_comparison, fig2_target_data_effect,
               fig3_domain_alignment, fig4_dann_discriminator,
               fig5_subject_variability, fig6_confusion_matrices):
        try:
            path = fn(tables["summary"] if fn in (
                fig1_method_comparison, fig2_target_data_effect) else
                tables["alignment"] if fn in (
                    fig3_domain_alignment, fig4_dann_discriminator) else
                tables["subject_level"], out_dir)
            if path:
                paths.append(path)
        except Exception as exc:
            print(f"  SKIP {fn.__name__}: {exc}")
    return paths
