"""Shared network used by all conditions (paper section 4.6).

F  : feature extractor  -> z
C  : class classifier   -> motor imagery logits
D  : domain discriminator -> source/target logits (DANN only, preceded by GRL)

Source-only / CORAL / MMD use F + C and differ only in their training loss.
DANN additionally uses D with a gradient reversal layer.
"""

import torch.nn as nn

from .config import UDAConfig
from .losses import GradientReversalLayer


class UDANet(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, cfg: UDAConfig,
                 use_domain: bool = False):
        super().__init__()
        self.use_domain = bool(use_domain)

        layers = []
        dim = int(input_dim)
        for hidden in cfg.hidden_dims:
            layers.extend([
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ])
            dim = hidden
        layers.extend([
            nn.Linear(dim, cfg.feature_dim),
            nn.ReLU(),
        ])
        self.feature = nn.Sequential(*layers)
        self.classifier = nn.Linear(cfg.feature_dim, int(n_classes))

        self.grl = GradientReversalLayer(alpha=1.0)
        self.domain_head = None
        if self.use_domain:
            self.domain_head = nn.Sequential(
                self.grl,
                nn.Linear(cfg.feature_dim, cfg.feature_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.feature_dim, 2),
            )

    def extract(self, x):
        """Return learned representation z (used for alignment metrics)."""
        return self.feature(x)

    def forward(self, x, return_domain: bool = False):
        z = self.feature(x)
        logits = self.classifier(z)
        domain_logits = None
        if return_domain:
            if self.domain_head is None:
                raise RuntimeError("Domain head only exists for the DANN model.")
            domain_logits = self.domain_head(z)
        return logits, z, domain_logits


def build_model(input_dim: int, n_classes: int, cfg: UDAConfig,
                method: str) -> UDANet:
    """Factory: one architecture per method, identical F/C capacity."""
    return UDANet(
        input_dim=input_dim,
        n_classes=n_classes,
        cfg=cfg,
        use_domain=(method == "dann"),
    )
