"""Non-training metrics and statistics shared by the evaluation scripts."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def latent_drift(latent: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Mean squared drift from the frozen reference representation."""
    return F.mse_loss(latent, reference)


def cosine_similarity(latent: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(latent, reference, dim=-1).mean()


def token_cosine_similarity(
    z_pred: torch.Tensor, z_target: torch.Tensor, layernorm: bool = False
) -> torch.Tensor:
    """Per-token cosine similarity [..., D] -> [...], optionally after channel-wise LayerNorm (FP32)."""
    z_pred, z_target = z_pred.float(), z_target.float()
    if layernorm:
        dim = z_pred.shape[-1]
        z_pred = F.layer_norm(z_pred, (dim,), eps=1e-6)
        z_target = F.layer_norm(z_target, (dim,), eps=1e-6)
    cosine = F.cosine_similarity(z_pred, z_target, dim=-1, eps=1e-8)
    return torch.nan_to_num(cosine, nan=0.0, posinf=1.0, neginf=-1.0)


def relative_feature_change(
    z_fused: torch.Tensor,
    z_static: torch.Tensor,
    mask_indices: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """||z_fused[mask] - z_static[mask]|| / (||z_static[mask]|| + eps) on masked tokens.

    z_fused / z_static: [B, T-1, N_p, D]; mask_indices: [B, M]. Returns [B, M].
    """
    batch, transitions, patches, dim = z_fused.shape
    fused = z_fused.float().reshape(batch, transitions * patches, dim)
    static = z_static.float().reshape(batch, transitions * patches, dim)
    index = mask_indices.clamp(0, fused.shape[1] - 1).unsqueeze(-1).expand(-1, -1, dim)
    fused, static = torch.gather(fused, 1, index), torch.gather(static, 1, index)
    delta = torch.linalg.vector_norm(fused - static, dim=-1) / (torch.linalg.vector_norm(static, dim=-1) + eps)
    return torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)


def compute_ade_fde(
    pred_traj: torch.Tensor, gt_traj: torch.Tensor, fps: float = 10.0
) -> tuple[dict[str, float], np.ndarray]:
    """ADE/FDE (meters) overall and at 1s/2s/3s horizons; pred/gt: [N, T, 2].

    Returns the metrics dict and the per-window ADE array used for confidence intervals.
    """
    errors = torch.linalg.vector_norm(pred_traj - gt_traj, dim=-1)  # [N, T]
    metrics = {"ADE": errors.mean().item(), "FDE": errors[:, -1].mean().item()}
    for seconds in (1, 2, 3):
        step = int(round(seconds * fps))
        if step <= errors.shape[1]:
            metrics[f"ADE_{seconds}s"] = errors[:, :step].mean().item()
            metrics[f"FDE_{seconds}s"] = errors[:, step - 1].mean().item()
    return metrics, errors.mean(dim=1).numpy()


def summarize_values(values: Sequence[float]) -> dict[str, float | int | None]:
    """Mean / std / median over finite values, with ``None`` entries when empty."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"mean": None, "std": None, "median": None, "n": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "n": int(len(array)),
    }


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    confidence: float = 0.95,
    block_size: int | None = 1,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Circular block bootstrap CI on the mean; ``block_size=1`` is the plain bootstrap.

    ``block_size=None`` selects ceil(n ** (1/3)).
    """
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    count = len(array)
    if count == 0:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "se": None, "n": 0}
    if count < 2:
        value = float(array[0])
        return {"mean": value, "ci_lower": value, "ci_upper": value, "se": 0.0, "n": count}

    if block_size is None:
        block_size = int(np.ceil(count ** (1.0 / 3.0)))
    block_size = max(1, min(block_size, count))
    rng = np.random.default_rng(seed)
    wrapped = np.concatenate([array, array[: block_size - 1]])
    n_blocks = int(np.ceil(count / block_size))
    starts = rng.integers(0, count, size=(n_boot, n_blocks))
    indices = (starts[..., None] + np.arange(block_size)).reshape(n_boot, -1)[:, :count]
    means = wrapped[indices].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0 * 100.0
    return {
        "mean": float(array.mean()),
        "ci_lower": float(np.percentile(means, alpha)),
        "ci_upper": float(np.percentile(means, 100.0 - alpha)),
        "se": float(means.std(ddof=1)),
        "n": int(count),
    }


def paired_bootstrap_ci(
    values_a: Sequence[float], values_b: Sequence[float], **kwargs
) -> dict[str, float | int | None]:
    """Bootstrap CI on the paired difference ``a - b`` (truncated to the shorter sequence)."""
    count = min(len(values_a), len(values_b))
    delta = np.asarray(values_a[:count], dtype=np.float64) - np.asarray(values_b[:count], dtype=np.float64)
    return bootstrap_ci(delta, **kwargs)
