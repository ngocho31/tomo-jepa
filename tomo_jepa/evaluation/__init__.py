"""Losses, metrics and evaluation scripts.

Evaluation scripts (``eval_*.py``) share the harness in :mod:`tomo_jepa.evaluation.common`
and are run as ``python -m tomo_jepa.evaluation.<script>``.
"""

from .hardware_monitor import HardwareTelemetryMonitor
from .losses import (
    ToMoJEPALoss,
    compute_jepa_loss,
    compute_orthogonality_loss,
)
from .metrics import (
    bootstrap_ci,
    compute_ade_fde,
    cosine_similarity,
    latent_drift,
    paired_bootstrap_ci,
    relative_feature_change,
    summarize_values,
    token_cosine_similarity,
)

__all__ = [
    "HardwareTelemetryMonitor",
    "ToMoJEPALoss",
    "bootstrap_ci",
    "compute_ade_fde",
    "compute_jepa_loss",
    "compute_orthogonality_loss",
    "cosine_similarity",
    "latent_drift",
    "paired_bootstrap_ci",
    "relative_feature_change",
    "summarize_values",
    "token_cosine_similarity",
]
