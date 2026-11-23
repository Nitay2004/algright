"""Metrics from paper sections 4.8 and 5.5.

Classification: accuracy, balanced accuracy, macro precision/recall/F1 and
Cohen's kappa on the held-out target test partition.

Alignment: covariance discrepancy and kernel MMD computed from source trials
and *unlabelled* target adaptation trials (never the target test partition).
For DANN, domain-discriminator accuracy is reported separately.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)

from .config import UDAConfig


def classification_metrics(y_true, y_pred, n_classes=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = None
    if n_classes is not None:
        labels = list(range(int(n_classes)))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
    }


def covariance_discrepancy(z_source: np.ndarray, z_target: np.ndarray) -> float:
    """Squared Frobenius distance between covariance matrices (normalised)."""
    zs = np.atleast_2d(np.asarray(z_source, dtype=np.float64))
    zt = np.atleast_2d(np.asarray(z_target, dtype=np.float64))
    zs = zs - zs.mean(axis=0, keepdims=True)
    zt = zt - zt.mean(axis=0, keepdims=True)
    d = zs.shape[1]
    cs = (zs.T @ zs) / max(len(zs) - 1, 1) + 1e-4 * np.eye(d)
    ct = (zt.T @ zt) / max(len(zt) - 1, 1) + 1e-4 * np.eye(d)
    return float(np.sum((cs - ct) ** 2) / (d * d))


def mmd_numpy(z_source: np.ndarray, z_target: np.ndarray,
              sigmas=(0.1, 0.5, 1.0, 2.0)) -> float:
    """Multi-kernel RBF MMD^2 between two feature sets."""
    zs = np.atleast_2d(np.asarray(z_source, dtype=np.float64))
    zt = np.atleast_2d(np.asarray(z_target, dtype=np.float64))

    def sq_dist(a, b):
        dist = (a ** 2).sum(1, keepdims=True) + (b ** 2).sum(1, keepdims=True).T
        dist -= 2.0 * a @ b.T
        return np.clip(dist, 0.0, None)

    values = []
    for sigma in sigmas:
        denom = 2.0 * float(sigma) ** 2
        kss = np.exp(-sq_dist(zs, zs) / denom)
        ktt = np.exp(-sq_dist(zt, zt) / denom)
        kst = np.exp(-sq_dist(zs, zt) / denom)
        values.append(kss.mean() + ktt.mean() - 2.0 * kst.mean())
    return float(np.mean(values))


def representation_metrics(z_source, z_target, cfg: UDAConfig):
    return {
        "covariance_discrepancy": covariance_discrepancy(z_source, z_target),
        "mmd": mmd_numpy(z_source, z_target, sigmas=cfg.mmd_sigmas),
    }


@torch.no_grad()
def domain_discriminator_accuracy(model, features, cfg: UDAConfig) -> float:
    """Accuracy of D on F(source) vs F(target); lower means better alignment."""
    if model.domain_head is None:
        return float("nan")
    device = next(model.parameters()).device
    previous_alpha = model.grl.alpha
    model.grl.alpha = 0.0  # evaluation only: no gradient inversion needed
    try:
        model.eval()
        zs = model.extract(
            torch.as_tensor(np.asarray(features.source_train, dtype=np.float32),
                            device=device)
        )
        zt = model.extract(
            torch.as_tensor(np.asarray(features.target_adapt, dtype=np.float32),
                            device=device)
        )
        ds = model.domain_head(zs).argmax(dim=1).cpu().numpy()
        dt = model.domain_head(zt).argmax(dim=1).cpu().numpy()
        y = np.concatenate([np.zeros(len(ds)), np.ones(len(dt))])
        pred = np.concatenate([ds, dt])
        return float(accuracy_score(y, pred))
    finally:
        model.grl.alpha = previous_alpha
