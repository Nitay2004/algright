"""Command-line entry point for the paper-aligned UDA experiment.

Quick smoke test (no dataset download)::

    py -m uda_cross_subject.run_uda --mode synthetic

Real data, all four methods, full target-data availability study::

    py -m uda_cross_subject.run_uda --mode dataset --dataset bci2a \
        --subjects 1 2 3 4 5 --proportions 0.25 0.5 0.75 1.0 --epochs 80

Every supported dataset in one command (same behaviour as run_pipeline.py)::

    py -m uda_cross_subject.run_uda --mode dataset --dataset all \
        --subjects 1 2 3
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __package__ in (None, ""):
    from data_pipeline.dataset_fetcher import DATASETS
    from uda_cross_subject import reporting
    from uda_cross_subject.config import METHODS, UDAConfig
    from uda_cross_subject.data import load_raw_data, preprocess_trials
    from uda_cross_subject.evaluation import run_experiment, set_seed
else:
    from data_pipeline.dataset_fetcher import DATASETS
    from . import reporting
    from .config import METHODS, UDAConfig
    from .data import load_raw_data, preprocess_trials
    from .evaluation import run_experiment, set_seed


ALL_DATASETS = tuple(DATASETS)


def resolve_datasets(names):
    """Expand ``all`` and de-duplicate, preserving command-line order."""
    datasets = []
    for name in names:
        if name == "all":
            candidates = ALL_DATASETS
        elif name in DATASETS:
            candidates = (name,)
        else:
            raise ValueError(
                f"Unknown dataset '{name}'. Choose from: {list(ALL_DATASETS)} or 'all'"
            )
        for candidate in candidates:
            if candidate not in datasets:
                datasets.append(candidate)
    return datasets or list(ALL_DATASETS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unsupervised Domain Adaptation for cross-subject EEG MI "
                    "(source_only / CORAL / MMD / DANN)."
    )
    p.add_argument("--mode", choices=["synthetic", "dataset"], default="synthetic")
    p.add_argument("--dataset", nargs="+", default=["bci2a"],
                   help="One or more MOABB dataset keys, or 'all' for every "
                        "supported dataset (used with --mode dataset)")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--proportions", type=float, nargs="+", default=[1.0],
                   help="Unlabelled target adaptation fractions, e.g. 0.25 0.5 1.0")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--lambda-coral", type=float, default=1000.0)
    p.add_argument("--lambda-mmd", type=float, default=1.0)
    p.add_argument("--lambda-dann", type=float, default=1.0)
    p.add_argument("--mmd-sigmas", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])

    p.add_argument("--band", type=float, nargs=2, default=[8.0, 30.0])
    p.add_argument("--csp-components", type=int, default=4)
    p.add_argument("--target-test-ratio", type=float, default=0.2)
    p.add_argument("--source-val-ratio", type=float, default=0.2)
    p.add_argument("--target-split", choices=["stratified", "chronological"],
                   default="stratified")
    p.add_argument("--no-car", action="store_true")
    p.add_argument("--no-notch", action="store_true")

    p.add_argument("--synthetic-subjects", type=int, default=3)
    p.add_argument("--synthetic-trials", type=int, default=60)
    p.add_argument("--synthetic-channels", type=int, default=16)
    p.add_argument("--synthetic-samples", type=int, default=250)
    p.add_argument("--synthetic-sfreq", type=float, default=250.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="results_uda")
    p.add_argument("--quiet", action="store_true")
    return p


def config_from_args(args, dataset_name=None) -> UDAConfig:
    name = dataset_name
    if name is None:
        name = args.dataset[0] if isinstance(args.dataset, list) else args.dataset
    return UDAConfig(
        mode=args.mode,
        dataset_name=name,
        subjects=args.subjects,
        methods=args.methods,
        target_proportions=args.proportions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        feature_dim=args.feature_dim,
        lambda_coral=args.lambda_coral,
        lambda_mmd=args.lambda_mmd,
        lambda_dann=args.lambda_dann,
        mmd_sigmas=tuple(args.mmd_sigmas),
        band=tuple(args.band),
        csp_components=args.csp_components,
        target_test_ratio=args.target_test_ratio,
        source_val_ratio=args.source_val_ratio,
        target_split_mode=args.target_split,
        use_car=not args.no_car,
        use_notch=not args.no_notch,
        synthetic_subjects=args.synthetic_subjects,
        synthetic_trials_per_subject=args.synthetic_trials,
        synthetic_channels=args.synthetic_channels,
        synthetic_samples=args.synthetic_samples,
        synthetic_sfreq=args.synthetic_sfreq,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    base_cfg = config_from_args(args)

    if args.mode == "dataset":
        datasets = resolve_datasets(args.dataset)
    else:
        # Synthetic mode ignores --dataset; keep a single config.
        datasets = ["synthetic"]

    succeeded, failed = [], []
    for index, dataset_name in enumerate(datasets):
        cfg = replace(base_cfg, dataset_name=dataset_name)
        set_seed(cfg.seed)

        print("\n" + "=" * 72)
        print(f"  DATASET {index + 1}/{len(datasets)}: {dataset_name} "
              f"| subjects: {cfg.subjects}")
        print("=" * 72)

        try:
            print("[UDA] Loading data ...")
            X, y, meta = load_raw_data(cfg)
            print(f"[UDA] Preprocessing {X.shape[0]} trials "
                  f"({X.shape[1]} channels @ {meta['sfreq']} Hz) ...")
            X = preprocess_trials(X, meta["sfreq"], cfg)
            last = index == len(datasets) - 1
            run_experiment(X, y, meta, cfg, write_combined=last)
            succeeded.append(dataset_name)
        except Exception as exc:
            failed.append(dataset_name)
            print(f"[UDA] FAILED on '{dataset_name}': {type(exc).__name__}: {exc}")
            if len(datasets) == 1:
                raise  # a single explicit dataset should fail loudly

    # One combined workbook containing every successful dataset's run folder.
    if len(datasets) > 1:
        combined = reporting.write_combined_workbook(cfg.output_dir)
        if combined:
            print(f"[UDA] Combined Excel (all datasets): {combined}")

    print("\n[UDA] DATASET SUMMARY")
    print(f"  Completed : {', '.join(succeeded) if succeeded else 'none'}")
    print(f"  Failed    : {', '.join(failed) if failed else 'none'}")
    return 1 if not succeeded else (2 if failed else 0)


if __name__ == "__main__":
    raise SystemExit(main())
