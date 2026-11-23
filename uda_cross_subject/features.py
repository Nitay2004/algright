"""Feature preparation shared by all four UDA conditions.

FBCSP is fitted on *source labelled trials only*.  The scaler is fitted on
source training features plus the unlabelled target adaptation features; the
held-out target test partition is never used for fitting (paper 4.3/4.7).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classification.feature_extraction import FBCSP  # noqa: E402
from preprocessing.baseline_filters import apply_bandpass  # noqa: E402

from .config import UDAConfig  # noqa: E402
from .data import DomainFold  # noqa: E402


@dataclass
class FeatureSet:
    source_train: np.ndarray
    source_train_labels: np.ndarray
    source_val: np.ndarray
    source_val_labels: np.ndarray
    target_adapt: np.ndarray
    target_test: np.ndarray
    kind: str


class LogPowerFeatures:
    """Label-free fallback: log band power per channel (CSP-free)."""

    def __init__(self, sfreq: float, bands):
        self.sfreq = sfreq
        self.bands = list(bands)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        parts = []
        for low, high in self.bands:
            high = min(high, 0.95 * self.sfreq / 2.0)
            filtered = apply_bandpass(
                np.asarray(X, dtype=np.float64),
                lowcut=low, highcut=high, sfreq=self.sfreq, order=4,
            )
            parts.append(np.log(np.mean(filtered ** 2, axis=-1) + 1e-10))
        return np.hstack(parts)


class FeatureBuilder:
    """Fit feature extraction and scaling without touching target test data."""

    def __init__(self, cfg: UDAConfig, sfreq: float):
        self.cfg = cfg
        self.sfreq = float(sfreq)
        self.transformer_ = None
        self.scaler_ = StandardScaler()
        self.kind = cfg.feature_kind

    def _build_transformer(self, X_source, y_source):
        if self.cfg.feature_kind == "fbcsp":
            try:
                n_components = max(1, min(
                    self.cfg.csp_components, X_source.shape[1] - 1,
                    len(np.unique(y_source)) * 2,
                ))
                transformer = FBCSP(
                    n_components=n_components,
                    sfreq=self.sfreq,
                    bands=list(self.cfg.fbcsp_bands),
                )
                # MNE prints rank/covariance diagnostics at INFO level.
                previous_level = mne.utils.logger.level
                try:
                    mne.set_log_level("ERROR")
                    transformer.fit(X_source, y_source)
                finally:
                    mne.set_log_level(previous_level)
                return transformer, "fbcsp"
            except Exception as exc:  # tiny/unusual datasets -> safe fallback
                if self.cfg.verbose:
                    print(f"[UDA] FBCSP unavailable ({exc}); using log-power features.")
        return LogPowerFeatures(self.sfreq, self.cfg.logpower_bands), "logpower"

    def fit_transform(self, fold: DomainFold) -> FeatureSet:
        self.transformer_, self.kind = self._build_transformer(
            fold.source_train, fold.source_train_labels
        )
        transform = self.transformer_.transform

        zs = np.asarray(transform(fold.source_train), dtype=np.float64)
        zv = np.asarray(transform(fold.source_val), dtype=np.float64)
        zt = np.asarray(transform(fold.target_adapt), dtype=np.float64)
        zte = np.asarray(transform(fold.target_test), dtype=np.float64)

        # Unsupervised target adaptation statistics are allowed; test is not.
        self.scaler_.fit(np.vstack([zs, zt]))

        def scale(arr):
            if len(arr) == 0:
                return arr.reshape(0, zs.shape[1] if zs.ndim == 2 else 0)
            return np.clip(self.scaler_.transform(arr), -10.0, 10.0).astype(np.float32)

        return FeatureSet(
            source_train=scale(zs),
            source_train_labels=np.asarray(fold.source_train_labels, dtype=np.int64),
            source_val=scale(zv),
            source_val_labels=np.asarray(fold.source_val_labels, dtype=np.int64),
            target_adapt=scale(zt),
            target_test=scale(zte),
            kind=self.kind,
        )
