"""Data loading, preprocessing and LOSO source/target construction.

Implements paper sections 4.2, 4.3 and 4.7:

* source domain = labelled trials from all subjects except the target,
* target adaptation domain = unlabelled trials of the held-out subject,
* target test domain = a separate held-out partition of that same subject.

Existing project modules are reused wherever possible (dataset loader,
bandpass / notch / CAR filters and the synthetic EEG generator).
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Allow both `python -m uda_cross_subject.run_uda` and direct script execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.dataset_fetcher import fetch_eeg_dataset  # noqa: E402
from preprocessing.baseline_filters import (  # noqa: E402
    apply_bandpass,
    apply_car,
    apply_notch,
)
from synthetic.eeg_generator import generate_synthetic_eeg  # noqa: E402

from .config import UDAConfig  # noqa: E402


@dataclass
class DomainFold:
    """One LOSO round for one target subject."""

    target_subject: int
    source_train: np.ndarray
    source_train_labels: np.ndarray
    source_val: np.ndarray
    source_val_labels: np.ndarray
    target_adapt: np.ndarray
    target_test: np.ndarray
    target_test_labels: np.ndarray
    n_classes: int

    @property
    def n_source(self) -> int:
        return len(self.source_train) + len(self.source_val)

    @property
    def n_target_adapt(self) -> int:
        return len(self.target_adapt)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def generate_synthetic_data(cfg: UDAConfig):
    """Build a small multi-subject binary MI dataset with a subject shift.

    Each synthetic "subject" has a different amplitude scale and noise level
    so that a genuine domain shift exists between source and target.  This is
    only meant for smoke-testing the framework, not for reporting results.
    """
    X_parts, y_parts, s_parts = [], [], []
    rng = np.random.RandomState(cfg.seed)

    for subject in range(1, cfg.synthetic_subjects + 1):
        # generate_synthetic_eeg uses the global numpy RNG.
        np.random.seed(cfg.seed + subject)
        labels = rng.randint(0, 2, size=cfg.synthetic_trials_per_subject)
        trials = np.stack([
            generate_synthetic_eeg(
                n_channels=cfg.synthetic_channels,
                n_samples=cfg.synthetic_samples,
                sfreq=cfg.synthetic_sfreq,
                class_label=int(label),
            )
            for label in labels
        ]).astype(np.float64)

        # Inter-subject domain shift (amplitude + measurement noise).
        scale = 1.0 + 0.35 * (subject - 1)
        trials = trials * scale + rng.normal(
            loc=0.0, scale=1.5 * (subject - 1), size=trials.shape
        )

        X_parts.append(trials)
        y_parts.append(labels)
        s_parts.append(np.full(len(labels), subject, dtype=int))

    meta = {
        "sfreq": cfg.synthetic_sfreq,
        "n_channels": cfg.synthetic_channels,
        "dataset_name": "synthetic",
        "subjects": np.concatenate(s_parts),
    }
    return np.concatenate(X_parts), np.concatenate(y_parts), meta


def load_raw_data(cfg: UDAConfig):
    """Return raw trials, labels and metadata for the configured mode."""
    if cfg.mode == "synthetic":
        return generate_synthetic_data(cfg)

    if cfg.mode == "dataset":
        X, y, meta = fetch_eeg_dataset(
            dataset_name=cfg.dataset_name,
            subjects=list(cfg.subjects),
            tmin=cfg.tmin,
            tmax=cfg.tmax,
            n_classes=cfg.n_classes,
        )
        return X.astype(np.float64), np.asarray(y), meta

    raise ValueError(f"Unknown mode '{cfg.mode}'. Use 'synthetic' or 'dataset'.")


# ----------------------------------------------------------------------
# Preprocessing (identical for source and target - paper section 4.3)
# ----------------------------------------------------------------------
def preprocess_trials(X: np.ndarray, sfreq: float, cfg: UDAConfig) -> np.ndarray:
    """Band-pass (8-30 Hz), optional notch, optional CAR.

    Parameters are estimated from the signal itself only - no target test
    statistics are used here.
    """
    Xf = np.asarray(X, dtype=np.float64)
    if cfg.use_car:
        Xf = apply_car(Xf)

    low, high = cfg.band
    nyq = sfreq / 2.0
    high = min(high, 0.95 * nyq)
    low = min(low, high - 1.0)
    Xf = apply_bandpass(Xf, lowcut=low, highcut=high, sfreq=sfreq, order=4)

    if cfg.use_notch and cfg.notch_freq < nyq:
        Xf = apply_notch(Xf, freq=cfg.notch_freq, sfreq=sfreq, quality=30.0)

    return np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------------------------------------------------
# LOSO domain construction
# ----------------------------------------------------------------------
def _split_indices(indices, labels, test_size, seed):
    """Stratified split with a safe fallback for very small classes."""
    if len(indices) < 4:
        cut = max(1, int(round(len(indices) * (1.0 - test_size))))
        return indices[:cut], indices[cut:]

    try:
        return train_test_split(
            indices,
            test_size=test_size,
            stratify=labels,
            random_state=seed,
            shuffle=True,
        )
    except ValueError:
        return train_test_split(
            indices,
            test_size=test_size,
            stratify=None,
            random_state=seed,
            shuffle=True,
        )


def _chronological_target_split(target_idx, y, test_ratio):
    """Last `test_ratio` of each class -> test (reduces temporal leakage).

    The MOABB loader does not expose session IDs in its metadata, so this is
    the closest available approximation of session-aware partitioning.
    """
    adapt_parts, test_parts = [], []
    for cls in np.unique(y[target_idx]):
        cls_idx = target_idx[y[target_idx] == cls]
        # Trial indices are chronological within a subject's recording.
        cut = max(1, int(round(len(cls_idx) * (1.0 - test_ratio))))
        cut = min(cut, len(cls_idx) - 1)
        adapt_parts.append(cls_idx[:cut])
        test_parts.append(cls_idx[cut:])
    return np.concatenate(adapt_parts), np.concatenate(test_parts)


def make_loso_folds(X, y, meta, cfg: UDAConfig):
    """Yield one :class:`DomainFold` per eligible target subject."""
    subject_ids = np.asarray(meta["subjects"])
    unique_subjects = sorted(set(subject_ids.tolist()))

    for target in unique_subjects:
        src_idx = np.where(subject_ids != target)[0]
        tgt_idx = np.where(subject_ids == target)[0]
        if len(src_idx) < 8 or len(tgt_idx) < 6:
            if cfg.verbose:
                print(f"[UDA] Skipping subject {target}: too few trials "
                      f"(source={len(src_idx)}, target={len(tgt_idx)})")
            continue

        src_train, src_val = _split_indices(
            src_idx, y[src_idx], cfg.source_val_ratio, cfg.seed
        )
        if cfg.target_split_mode == "chronological":
            tgt_adapt, tgt_test = _chronological_target_split(
                tgt_idx, y, cfg.target_test_ratio
            )
        else:
            tgt_adapt, tgt_test = _split_indices(
                tgt_idx, y[tgt_idx], cfg.target_test_ratio, cfg.seed
            )

        fold = DomainFold(
            target_subject=int(target),
            source_train=X[src_train],
            source_train_labels=y[src_train],
            source_val=X[src_val],
            source_val_labels=y[src_val],
            target_adapt=X[tgt_adapt],
            target_test=X[tgt_test],
            target_test_labels=y[tgt_test],
            n_classes=int(len(np.unique(y[src_idx]))),
        )
        yield fold


def select_target_proportion(fold: DomainFold, proportion: float,
                             cfg: UDAConfig) -> DomainFold:
    """Keep a fixed random subset of *unlabelled* adaptation trials.

    Target labels are never consulted here (paper section 4.9).
    """
    if proportion >= 1.0:
        return fold

    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(fold.n_target_adapt)
    n_keep = max(1, int(round(proportion * len(perm))))
    return replace(fold, target_adapt=fold.target_adapt[perm[:n_keep]])
