"""
Build Excel workbooks summarising dataset runs produced by the pipeline.

Scans <base>/run_*/metrics.json files (created by run_pipeline.py in dataset
mode) and writes:

    1. Combined workbook (results_summary.xlsx):
         Sheet "Summary"        : one row per dataset run, key metrics
         Sheet "Stage Accuracy" : per-stage test accuracy + CV (pivoted)

    2. One individual workbook per dataset run (<dataset>_results.xlsx):
         Sheet "Overview"        : key metrics at a glance
         Sheet "Stage Accuracy"  : per-stage test accuracy + CV
         Sheet "Config"          : pipeline configuration used for the run
         Sheet "Metrics"         : every recorded metric

When called from run_pipeline.py only the run directories created during that
execution are included, so the sheets always mirror the last pipeline run.

Usage (standalone):
    py make_results_excel.py [--base results] [--out results_summary.xlsx]
"""
import argparse
import datetime
import glob
import json
import os
import re
from copy import copy

import pandas as pd

STAGE_ORDER = ["Noisy", "Bandpass+Notch", "LMS", "DAE", "Proposed Adaptive"]

KEY_METRICS = [
    "classification_accuracy", "precision", "recall", "f1_score",
    "corr_fidelity",
]

CONFIG_KEYS = [
    "dataset_name", "dataset_subjects", "sfreq", "n_channels", "epoch_samples",
    "l_freq", "h_freq", "notch_freq", "filter_type", "classifier",
    "svm_kernel", "use_fbcsp", "fbcsp_bands", "csp_n_components",
    "dae_blend_alpha", "dae_finetune", "dae_finetune_epochs", "epochs",
    "batch_size", "lr", "latent_dim", "use_car", "artifact_rejection_ratio",
    "noise_snr_db", "test_noise_shift_db",
]


def _is_dataset_run(path):
    """True only for pipeline runs in dataset mode (run_*_dataset_*)."""
    return bool(re.search(r"_dataset(?:_|$)",
                          os.path.basename(os.path.dirname(path))))


def find_metric_files(base, run_dirs=None):
    pattern = os.path.join(base, "run_*", "metrics.json")
    paths = sorted(glob.glob(pattern))
    paths = [p for p in paths if _is_dataset_run(p)]
    if run_dirs:
        allowed = {os.path.normcase(os.path.abspath(d)) for d in run_dirs}
        paths = [p for p in paths
                 if os.path.normcase(os.path.abspath(os.path.dirname(p)))
                 in allowed]
    return paths


def dataset_name_from_dir(path, config):
    if config and config.get("dataset_name"):
        return config["dataset_name"]
    m = re.search(r"_dataset(?:_(.+))?$", os.path.basename(os.path.dirname(path)))
    if m and m.group(1):
        return m.group(1)
    return os.path.basename(os.path.dirname(path))


def load_runs(base, run_dirs=None):
    rows = []
    for path in find_metric_files(base, run_dirs=run_dirs):
        payload = load_payload(path)
        if payload is None:
            continue
        config = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        stage_accs = metrics.get("stage_accs", {}) or {}
        stage_cv = metrics.get("stage_cv", {}) or {}

        entry = {
            "dataset": dataset_name_from_dir(path, config),
            "timestamp": payload.get("timestamp", ""),
            "subjects": config.get("dataset_subjects", []),
            "sfreq": config.get("sfreq"),
            "n_channels": config.get("n_channels"),
            "classifier": config.get("classifier", "auto"),
            "classification_accuracy": metrics.get("classification_accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "best_stage": metrics.get("best_stage"),
            "corr_fidelity": metrics.get("corr_fidelity"),
        }
        for stage in STAGE_ORDER:
            entry[f"acc_{stage}"] = stage_accs.get(stage)
            entry[f"cv_{stage}"] = stage_cv.get(stage)

        entry["source_dir"] = os.path.dirname(path)
        rows.append(entry)
    return rows


def load_payload(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Excel] Skipping unreadable {path}: {e}")
        return None


def build_dataframes(rows):
    summary_cols = [
        "dataset", "timestamp", "subjects", "sfreq", "n_channels",
        "classifier",
    ] + KEY_METRICS + ["best_stage"]
    summary_cols += [f"acc_{s}" for s in STAGE_ORDER]
    summary = pd.DataFrame(rows)[summary_cols]

    pivots = []
    for r in rows:
        for stage in STAGE_ORDER:
            acc = r.get(f"acc_{stage}")
            cv = r.get(f"cv_{stage}")
            if acc is None and cv is None:
                continue
            pivots.append({"dataset": r["dataset"], "stage": stage,
                           "test_acc": acc, "cv_acc": cv})
    stage_acc = pd.DataFrame(pivots)
    if not stage_acc.empty:
        stage_acc = stage_acc.sort_values(["dataset", "stage"])
    return summary, stage_acc


def _fmt(df):
    out = df.copy()
    for col in out.columns:
        if col.startswith(("acc_", "cv_")) or col in KEY_METRICS:
            out[col] = out[col].apply(
                lambda v: "" if v is None else round(float(v), 4))
        if col == "subjects":
            out[col] = out[col].apply(lambda v: str(v))
    return out


def _style_header(ws):
    for cell in ws[1]:
        f = copy(cell.font)
        f.bold = True
        cell.font = f
    for col_cells in ws.columns:
        max_len = max(len(str(c.value)) if c.value is not None else 0
                      for c in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = max_len + 2


def _value_str(v):
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4f}"
    return v


def _run_timestamp(run_dir, payload=None):
    """Timestamp token for an individual workbook (unique per run folder)."""
    m = re.search(r"run_(\d{8}_\d{6})", os.path.basename(run_dir))
    if m:
        return m.group(1)
    ts = (payload or {}).get("timestamp", "")
    if ts:
        return re.sub(r"[^0-9A-Za-z]+", "_", ts).strip("_")
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_run_sheets(writer, payload, run_dir, prefix=""):
    """Write a run's detail sheets (Overview/Stage Accuracy/Config/Metrics).

    Sheets are named ``<prefix> - <name>`` (Excel max 31 chars respected).
    """
    config = payload.get("config", {}) or {}
    metrics = payload.get("metrics", {}) or {}
    stage_accs = metrics.get("stage_accs", {}) or {}
    stage_cv = metrics.get("stage_cv", {}) or {}
    dataset = dataset_name_from_dir(run_dir, config)

    overview = [
        ("Dataset", dataset),
        ("Timestamp", payload.get("timestamp", "")),
        ("Best Stage", metrics.get("best_stage", "")),
        ("Source", run_dir),
    ]
    for k in KEY_METRICS:
        overview.append((k.replace("_", " ").title(), metrics.get(k)))
    overview.append(("", ""))
    overview.append(("Stage Accuracies", ""))
    for stage in STAGE_ORDER:
        acc = stage_accs.get(stage)
        cv = stage_cv.get(stage)
        if acc is not None or cv is not None:
            acc_txt = "" if acc is None else f"{acc:.4f}"
            cv_txt = "" if cv is None else f"{cv:.4f}"
            overview.append((stage, f"test={acc_txt} | cv={cv_txt}"))

    ov_df = pd.DataFrame(overview, columns=["Metric", "Value"])

    pivots = []
    for stage in STAGE_ORDER:
        acc = stage_accs.get(stage)
        cv = stage_cv.get(stage)
        if acc is None and cv is None:
            continue
        pivots.append({"Stage": stage, "Test Accuracy": acc, "CV Accuracy": cv})
    stage_df = pd.DataFrame(pivots)

    config_rows = [(k, _value_str(v)) for k, v in config.items()
                   if k in CONFIG_KEYS]
    config_df = pd.DataFrame(config_rows, columns=["Config", "Value"])

    metrics_rows = []
    for k, v in metrics.items():
        if k in ("stage_accs", "stage_cv"):
            continue
        metrics_rows.append((k, _value_str(v)))
    metrics_df = pd.DataFrame(metrics_rows, columns=["Metric", "Value"])

    def sheet(name):
        full = f"{prefix} - {name}".strip(" - ") if prefix else name
        if len(full) > 31:
            full = full[:31]
        return full

    ov_df.to_excel(writer, sheet_name=sheet("Overview"), index=False)
    _style_header(writer.sheets[sheet("Overview")])
    _fmt(stage_df).to_excel(writer, sheet_name=sheet("Stage Accuracy"),
                            index=False)
    _style_header(writer.sheets[sheet("Stage Accuracy")])
    config_df.to_excel(writer, sheet_name=sheet("Config"), index=False)
    _style_header(writer.sheets[sheet("Config")])
    metrics_df.to_excel(writer, sheet_name=sheet("Metrics"), index=False)
    _style_header(writer.sheets[sheet("Metrics")])


def write_single_run_excel(payload, out_path):
    """Write one rich workbook for a single dataset run (individual Excel)."""
    dataset = dataset_name_from_dir(out_path, payload.get("config", {}) or {})
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _write_run_sheets(writer, payload, os.path.dirname(out_path))

    print(f"[Excel] Individual: {os.path.abspath(out_path)}")


def build_excel(base="results", out=None, run_dirs=None,
                individual=True):
    """Aggregate pipeline metrics into Excel workbook(s).

    Args:
        base: directory containing run_* subfolders (default "results")
        out: output .xlsx path for the combined workbook; when None a fresh
            timestamped file (results_summary_YYYYmmdd_HHMMSS.xlsx) is created
            inside ``base`` on every call so past runs are never overwritten
        run_dirs: optional iterable of run directories to include; when given,
            only metrics.json from exactly these runs are used (run_pipeline.py
            passes the runs it just created so the sheets mirror the last run)
        individual: also write one Excel per dataset run into <base>

    Returns:
        tuple (str, int): the combined workbook path and the number of dataset
        runs written
    """
    files = find_metric_files(base, run_dirs=run_dirs)
    rows = load_runs(base, run_dirs=run_dirs)
    if not rows:
        msg = f"No metrics.json found under '{base}'."
        if run_dirs:
            msg += " (no fresh results from this run)"
        print(f"[Excel] {msg}")
        return "", 0

    if out is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(base, f"results_summary_{stamp}.xlsx")

    summary, stage_acc = build_dataframes(rows)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        _fmt(summary).to_excel(writer, sheet_name="Summary", index=False)
        _style_header(writer.sheets["Summary"])
        _fmt(stage_acc).to_excel(writer, sheet_name="Stage Accuracy", index=False)
        _style_header(writer.sheets["Stage Accuracy"])

    print(f"[Excel] Combined: {os.path.abspath(out)} "
          f"({len(rows)} dataset run(s))")

    if individual:
        for path in files:
            payload = load_payload(path)
            if payload is None:
                continue
            config = payload.get("config", {}) or {}
            dset = dataset_name_from_dir(path, config)
            run_dir = os.path.dirname(path)
            stamp = _run_timestamp(run_dir, payload)
            ind_path = os.path.join(base, f"{dset}_results_{stamp}.xlsx")
            try:
                write_single_run_excel(payload, ind_path)
            except PermissionError:
                print(f"[Excel] SKIP (file open in Excel?): {ind_path}")

    return out, len(rows)


def build_master_excel(base="results", out=None, run_dirs=None):
    """Write one consolidated workbook containing summary + every run's detail.

    Sheets:
        "Summary"             – one row per dataset
        "Stage Accuracy"      – per-stage test/CV (all datasets pivoted)
        "<dataset> Overview"  – per-run key metrics
        "<dataset> Stage Acc" – per-run stage breakdown
        "<dataset> Config"    – pipeline config for the run
        "<dataset> Metrics"   – every recorded metric

    Returns (out_path, n_runs).
    """
    files = find_metric_files(base, run_dirs=run_dirs)
    rows = load_runs(base, run_dirs=run_dirs)
    if not rows:
        msg = f"No metrics.json found under '{base}'."
        if run_dirs:
            msg += " (no fresh results from this run)"
        print(f"[Excel] {msg}")
        return "", 0

    if out is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(base, f"results_all_{stamp}.xlsx")

    summary, stage_acc = build_dataframes(rows)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        _fmt(summary).to_excel(writer, sheet_name="Summary", index=False)
        _style_header(writer.sheets["Summary"])
        _fmt(stage_acc).to_excel(writer, sheet_name="Stage Accuracy", index=False)
        _style_header(writer.sheets["Stage Accuracy"])

        for path in files:
            payload = load_payload(path)
            if payload is None:
                continue
            config = payload.get("config", {}) or {}
            dset = dataset_name_from_dir(path, config)
            _write_run_sheets(writer, payload, os.path.dirname(path), prefix=dset)

    print(f"[Excel] Master: {os.path.abspath(out)} "
          f"({len(rows)} dataset run(s))")
    return out, len(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate pipeline results into Excel")
    parser.add_argument("--base", default="results", help="Results directory")
    parser.add_argument("--out", default=None,
                        help="Combined workbook path "
                             "(default: fresh results_summary_<timestamp>.xlsx)")
    parser.add_argument("--no-individual", action="store_true",
                        help="Skip per-dataset Excel files")
    parser.add_argument("--master", action="store_true",
                        help="Write a single workbook with Summary + all "
                             "per-dataset detail sheets")
    args = parser.parse_args()

    if args.master:
        out, written = build_master_excel(args.base, args.out)
    else:
        out, written = build_excel(args.base, args.out,
                                   individual=not args.no_individual)
    return 0 if written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())