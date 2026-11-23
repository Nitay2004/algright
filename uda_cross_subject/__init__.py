"""Paper-aligned Unsupervised Domain Adaptation (UDA) framework.

Reproduces the experimental framework of:

    "Unsupervised Domain Adaptation for Cross-Subject Generalization in
     EEG-Based Brain-Computer Interfaces: A Multi-Dataset Study"

Four conditions are implemented: source_only (baseline), CORAL, MMD and DANN,
all evaluated with a Leave-One-Subject-Out (LOSO) protocol.
"""

from .config import UDAConfig, METHODS

__all__ = ["UDAConfig", "METHODS"]
