"""Training loops for the four experimental conditions.

All methods see exactly the same source labels and the same unlabelled target
adaptation trials.  Only the loss differs:

    source_only : L = L_cls
    coral       : L = L_cls + lambda_CORAL * L_CORAL
    mmd         : L = L_cls + lambda_MMD   * L_MMD
    dann        : L = L_cls + lambda_DANN  * L_domain  (through gradient reversal)

Model selection uses source-domain validation only - target test labels are
never accessed here (paper sections 4.4 - 4.6, 5.3).
"""

from copy import deepcopy
import math

import numpy as np
import torch
import torch.nn.functional as F_torch

from .config import UDAConfig
from .features import FeatureSet
from .losses import coral_loss, mmd_loss
from .models import UDANet


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class UDATrainer:
    def __init__(self, model: UDANet, cfg: UDAConfig, method: str):
        if method not in ("source_only", "coral", "mmd", "dann"):
            raise ValueError(f"Unknown method: {method}")
        self.model = model.to(resolve_device(cfg.device))
        self.cfg = cfg
        self.method = method
        self.device = next(self.model.parameters()).device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    def _grl_lambda(self, epoch: int, total_epochs: int) -> float:
        """Standard DANN schedule: lambda = 2/(1+exp(-10p)) - 1."""
        if not self.cfg.use_grl_schedule:
            return 1.0
        p = epoch / max(total_epochs - 1, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

    def _batch_loss(self, xs, ys, xt, epoch, total_epochs):
        is_dann = self.method == "dann"
        if is_dann:
            self.model.grl.alpha = self._grl_lambda(epoch, total_epochs)

        logits_s, zs, domain_s = self.model(xs, return_domain=is_dann)
        loss_cls = self.criterion(logits_s, ys)

        # Source-only: no target data is used at all.
        if self.method == "source_only" or xt is None or len(xt) == 0:
            return loss_cls

        target_idx = torch.randint(
            0, len(xt), (len(xs),), device=self.device
        )
        xt_batch = xt[target_idx]

        if self.method == "coral":
            _, zt, _ = self.model(xt_batch)
            return loss_cls + self.cfg.lambda_coral * coral_loss(zs, zt)

        if self.method == "mmd":
            _, zt, _ = self.model(xt_batch)
            return loss_cls + self.cfg.lambda_mmd * mmd_loss(
                zs, zt, sigmas=self.cfg.mmd_sigmas, kernel=self.cfg.mmd_kernel
            )

        # DANN: domain discriminator sees both domains with source=0, target=1.
        _, zt, domain_t = self.model(xt_batch, return_domain=True)
        zeros = torch.zeros(len(xs), dtype=torch.long, device=self.device)
        ones = torch.ones(len(xt_batch), dtype=torch.long, device=self.device)
        loss_domain = (
            self.criterion(domain_s, zeros) + self.criterion(domain_t, ones)
        )
        return loss_cls + self.cfg.lambda_dann * loss_domain

    @torch.no_grad()
    def _validation_loss(self, xv, yv, fallback):
        if xv is None or len(xv) == 0:
            return float(fallback)
        self.model.eval()
        logits, _, _ = self.model(xv)
        return float(self.criterion(logits, yv))

    # ------------------------------------------------------------------
    def fit(self, features: FeatureSet):
        xs = torch.as_tensor(features.source_train, device=self.device)
        ys = torch.as_tensor(features.source_train_labels, device=self.device,
                             dtype=torch.long)
        xv = torch.as_tensor(features.source_val, device=self.device)
        yv = torch.as_tensor(features.source_val_labels, device=self.device,
                             dtype=torch.long)
        xt = torch.as_tensor(features.target_adapt, device=self.device)

        history = {"train_loss": [], "val_loss": [], "grl_lambda": []}
        best_val = float("inf")
        best_state = deepcopy(self.model.state_dict())
        patience = 0
        epochs_run = 0

        for epoch in range(self.cfg.epochs):
            self.model.train()
            perm = torch.randperm(len(xs), device=self.device)
            running, batches = 0.0, 0

            for start in range(0, len(xs), self.cfg.batch_size):
                batch_idx = perm[start:start + self.cfg.batch_size]
                loss = self._batch_loss(
                    xs[batch_idx], ys[batch_idx], xt, epoch, self.cfg.epochs
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running += float(loss.detach())
                batches += 1

            train_loss = running / max(batches, 1)
            val_loss = self._validation_loss(xv, yv, train_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["grl_lambda"].append(
                self._grl_lambda(epoch, self.cfg.epochs) if self.method == "dann" else 0.0
            )
            epochs_run = epoch + 1

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= self.cfg.early_stop_patience:
                    break

        self.model.load_state_dict(best_state)
        history["epochs_run"] = epochs_run
        history["best_val_loss"] = best_val
        return history


@torch.no_grad()
def predict(model: UDANet, features: np.ndarray, cfg: UDAConfig):
    """Return (probabilities, predicted labels) for the given features."""
    device = next(model.parameters()).device
    model.eval()
    x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
    logits, _, _ = model(x)
    proba = F_torch.softmax(logits, dim=1).cpu().numpy()
    return proba, np.argmax(proba, axis=1)
