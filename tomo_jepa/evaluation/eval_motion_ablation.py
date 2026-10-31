"""Evaluate the contribution of motion features with controlled ablations.

The evaluation runs in two passes over the same deterministic windows. The
calibration pass computes a mean-position prediction baseline and patch-level
optical-flow magnitudes, which partition masked spatiotemporal tokens into
static and dynamic groups. The variant pass then evaluates the full ToMo-JEPA
model, a static-only variant, and a random-flow variant on identical masks.
Per-token latent changes between fused and static features are summarized by
group, with paired circular block-bootstrap confidence intervals across batches.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..contracts import FrameBatch
from ..pipeline import ToMoJEPAPipeline
from .common import autocast, build_pipeline, logger, make_eval_loader, run_evaluation, set_seed
from .metrics import paired_bootstrap_ci, relative_feature_change, summarize_values, token_cosine_similarity

TASK = "motion_ablation"
VARIANTS = ("full", "static_only", "random_flow")
SCORES = (
    "cos_global",
    "cos_dynamic",
    "cos_static",
    "baseline_gap_dynamic",
    "delta_motion_global",
    "delta_motion_dynamic",
    "delta_motion_static",
)


# ==========================================
# 1. FORWARD HOOKS DEFINING THE VARIANTS
# ==========================================

class FlowCaptureHook:
    """Forward hook on model.flow_estimator to capture optical flow without running FlowFormer++ twice."""

    def __init__(self) -> None:
        self.last_flow: torch.Tensor | None = None

    def __call__(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        if isinstance(output, torch.Tensor):
            self.last_flow = output.detach()


class ZeroFlowHook:
    """Zero out flow_encoder output."""

    def __call__(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(output)


class StaticOnlyFusionHook:
    """Bypass GatedCrossAttentionFusion so z_fused is strictly identical to z_static (inputs[0])."""

    def __call__(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        if inputs and isinstance(inputs[0], torch.Tensor):
            return inputs[0]
        return output


class RandomFlowHook:
    """Replace flow_encoder output with Gaussian noise N(mean, std^2 I)."""

    def __init__(self, base_seed: int, mean: float = 0.0, std: float = 1.0) -> None:
        self.base_seed = base_seed
        self.mean = mean
        self.std = std
        self.call_count = 0

    def __call__(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device=output.device)
        generator.manual_seed(self.base_seed + self.call_count)
        self.call_count += 1
        noise = torch.randn(output.shape, generator=generator, dtype=torch.float32, device=output.device)
        return (noise * self.std + self.mean).to(dtype=output.dtype)


# ==========================================
# 2. FLOW-BASED TOKEN PARTITION
# ==========================================

def reset_mask_sampler_counter(model: ToMoJEPAPipeline, value: int = -1) -> None:
    """Reset the _MaskGenerator iteration counter so every pass gets identical masks."""
    sampler = getattr(model, "mask_sampler", None)
    if sampler is None:
        return
    if hasattr(sampler, "_get_generator"):
        try:
            sampler._get_generator(max(1, model.config.history_length - 1))
        except Exception:
            pass
    for generator in getattr(sampler, "_generator_cache", {}).values():
        if hasattr(generator, "_itr_counter"):
            with generator._itr_counter.get_lock():
                generator._itr_counter.value = value


def infer_spatial_grid(model: ToMoJEPAPipeline, total_spatial_tokens: int) -> tuple[int, int]:
    """Infer (grid_h, grid_w) matching the ViT patch grid."""
    width, height = model.config.image_size
    patch_size = getattr(model.mask_sampler, "patch_size", 16)
    grid_h, grid_w = height // patch_size, width // patch_size
    if grid_h * grid_w == total_spatial_tokens and grid_h > 0 and grid_w > 0:
        return grid_h, grid_w
    side = int(math.isqrt(total_spatial_tokens))
    if side * side == total_spatial_tokens:
        return side, side
    for candidate in range(side, 0, -1):
        if total_spatial_tokens % candidate == 0:
            return candidate, total_spatial_tokens // candidate
    return 1, total_spatial_tokens


def compute_patch_flow_magnitudes(flow_maps: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Pool pixel-level optical flow [B, T-1, 2, H, W] into patch-level ||flow|| [B, (T-1)*N_p]."""
    if flow_maps.ndim != 5:
        raise ValueError(f"Expected 5D flow_maps [B, T-1, C, H, W], got shape {tuple(flow_maps.shape)}")
    batch, transitions, _, height, width = flow_maps.shape
    magnitude = torch.linalg.vector_norm(flow_maps.float(), dim=2)
    magnitude = torch.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0).reshape(batch * transitions, 1, height, width)
    pooled = F.adaptive_avg_pool2d(magnitude, (grid_h, grid_w))
    return pooled.reshape(batch, transitions * grid_h * grid_w)


def classify_masked_tokens_by_flow(
    mag_patches: torch.Tensor,
    mask_indices: torch.Tensor,
    top_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition masked tokens [B, M] into dynamic (top ``top_fraction`` by flow magnitude) and static masks.

    Per-sample top-k selection guarantees a valid split even for zero-flow frames.
    """
    batch, count = mask_indices.shape
    device = mask_indices.device
    if count == 0:
        empty = torch.zeros(batch, 0, dtype=torch.bool, device=device)
        return empty, empty

    mag_patches = mag_patches.to(device=device, dtype=torch.float32)
    if int(mask_indices.max().item()) >= mag_patches.shape[1]:
        mag_patches = F.interpolate(
            mag_patches.unsqueeze(1), size=int(mask_indices.max().item()) + 1, mode="linear", align_corners=False
        ).squeeze(1)
    magnitude = torch.gather(mag_patches, 1, mask_indices)

    if count == 1:
        dynamic = torch.ones(batch, 1, dtype=torch.bool, device=device)
        return dynamic, ~dynamic

    k = max(1, min(count - 1, int(round(top_fraction * count))))
    top_index = torch.topk(magnitude, k=k, dim=1).indices
    dynamic = torch.zeros(batch, count, dtype=torch.bool, device=device)
    dynamic.scatter_(1, top_index, True)
    return dynamic, ~dynamic


def flow_for_batch(model: ToMoJEPAPipeline, batch: FrameBatch, grid: tuple[int, int]) -> torch.Tensor:
    return compute_patch_flow_magnitudes(model.flow_estimator(batch.images), *grid)


# ==========================================
# 3. PASSES
# ==========================================

def prepare_pass(model: ToMoJEPAPipeline, seed: int) -> None:
    """Synchronize RNG and mask sampler so every pass sees the same masks."""
    model.eval()
    set_seed(seed)
    reset_mask_sampler_counter(model, value=-1)


@torch.no_grad()
def collect_calibration(model: ToMoJEPAPipeline, loader, args, device) -> dict[str, Any]:
    """Pass 1: per-patch flow magnitudes and the mean-position baseline z* on masked targets."""
    prepare_pass(model, args.seed)
    dim = model.config.latent_dim
    target_sum = torch.zeros(dim, device=device, dtype=torch.float64)
    target_count = 0
    position_sum: torch.Tensor | None = None
    position_count: torch.Tensor | None = None
    mag_patches: list[torch.Tensor] = []

    flow_hook = FlowCaptureHook()
    handle = model.flow_estimator.register_forward_hook(flow_hook)
    try:
        for batch in tqdm(loader, desc="Calibration"):
            flow_hook.last_flow = None
            batch.images = batch.images.to(device, non_blocking=True)
            with autocast(args, device):
                out = model(batch)
            flow_maps = flow_hook.last_flow
            if flow_maps is None:
                with autocast(args, device):
                    flow_maps = model.flow_estimator(batch.images)

            targets = out.target_patches.double()  # [B, M, D]
            target_sum += targets.sum(dim=(0, 1))
            target_count += targets.shape[0] * targets.shape[1]

            total_tokens = out.z_task.shape[1]
            if position_sum is None:
                position_sum = torch.zeros(total_tokens, dim, device=device, dtype=torch.float64)
                position_count = torch.zeros(total_tokens, 1, device=device, dtype=torch.float64)
            flat_index = out.mask_indices.reshape(-1).clamp(0, total_tokens - 1)
            position_sum.index_add_(0, flat_index, targets.reshape(-1, dim))
            position_count.index_add_(0, flat_index, torch.ones(len(flat_index), 1, device=device, dtype=torch.float64))

            grid = infer_spatial_grid(model, out.static_features.shape[2])
            mag_patches.append(compute_patch_flow_magnitudes(flow_maps, *grid).cpu())
    finally:
        handle.remove()

    global_mean = (target_sum / max(target_count, 1)).float()
    mean_position = torch.where(
        position_count > 0,
        position_sum / position_count.clamp(min=1.0),
        global_mean.double().unsqueeze(0),
    ).float()
    return {"mean_target": global_mean.cpu(), "mean_position_target": mean_position.cpu(), "mag_patches": mag_patches}


@torch.no_grad()
def evaluate_variant(
    model: ToMoJEPAPipeline,
    loader,
    name: str,
    calibration: dict[str, Any],
    args,
    device,
    hooks: list[tuple[nn.Module, Any]] | None = None,
) -> dict[str, Any]:
    """Pass 2: evaluate one variant on masked tokens, split into dynamic/static by flow magnitude."""
    handles = [module.register_forward_hook(hook) for module, hook in hooks or []]
    prepare_pass(model, args.seed)
    global_mean = calibration["mean_target"].to(device)
    mean_position = calibration["mean_position_target"].to(device)
    scores: dict[str, list[float]] = {key: [] for key in SCORES}
    counts = {"masked": 0, "dynamic": 0, "static": 0}

    try:
        for i, batch in enumerate(tqdm(loader, desc=f"Variant: {name}")):
            batch.images = batch.images.to(device, non_blocking=True)
            with autocast(args, device):
                out = model(batch)

            cosine = token_cosine_similarity(out.z_pred, out.target_patches, layernorm=True)  # [B, M]
            batch_size, num_masked = cosine.shape
            if num_masked == 0:
                continue

            mag = calibration["mag_patches"][i]
            if mag.shape[0] != batch_size:
                mag = flow_for_batch(model, batch, infer_spatial_grid(model, out.static_features.shape[2]))
            dynamic, static = classify_masked_tokens_by_flow(mag, out.mask_indices, args.top_fraction)

            scores["cos_global"].append(cosine.mean().item())
            counts["masked"] += batch_size * num_masked
            counts["dynamic"] += int(dynamic.sum())
            counts["static"] += int(static.sum())

            delta = relative_feature_change(out.fused_features, out.static_features, out.mask_indices)
            scores["delta_motion_global"].append(delta.mean().item())
            for region, region_mask in (("dynamic", dynamic), ("static", static)):
                if region_mask.any():
                    scores[f"cos_{region}"].append(cosine[region_mask].mean().item())
                    scores[f"delta_motion_{region}"].append(delta[region_mask].mean().item())

            if dynamic.any():
                dim = out.target_patches.shape[-1]
                if mean_position.shape[0] > int(out.mask_indices.max()):
                    index = out.mask_indices.unsqueeze(-1).expand(-1, -1, dim)
                    baseline = torch.gather(mean_position.unsqueeze(0).expand(batch_size, -1, -1), 1, index)
                else:
                    baseline = global_mean.view(1, 1, dim).expand_as(out.target_patches)
                baseline_cosine = token_cosine_similarity(baseline, out.target_patches, layernorm=True)
                scores["baseline_gap_dynamic"].append(
                    (cosine[dynamic].mean() - baseline_cosine[dynamic].mean()).item()
                )
    finally:
        for handle in handles:
            handle.remove()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"scores": scores, "counts": counts}


# ==========================================
# 4. REPORTING
# ==========================================

def generate_chart(summary: dict[str, Any], paired: dict[str, Any], chart_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(VARIANTS))
    width = 0.25
    regions = (("global", "Global", "#2196F3"), ("dynamic", "Dynamic (top flow)", "#FF9800"), ("static", "Static", "#4CAF50"))
    for offset, (key, label, color) in zip((-width, 0.0, width), regions):
        values = [summary[v][f"cos_{key}"]["mean"] or 0.0 for v in VARIANTS]
        axes[0].bar(x + offset, values, width, label=label, color=color)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(VARIANTS)
    axes[0].set_ylabel("LayerNorm cosine similarity")
    axes[0].set_title("Performance by region")
    axes[0].legend()

    items = [paired.get(f"cos_{key}_full_vs_static_only") for key, _, _ in regions]
    if all(item and item["mean"] is not None for item in items):
        means = [item["mean"] for item in items]
        errors = np.array([[item["mean"] - item["ci_lower"], item["ci_upper"] - item["mean"]] for item in items]).T
        axes[1].bar([label for _, label, _ in regions], means, yerr=errors, capsize=5, color=[c for _, _, c in regions])
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title("Paired delta (full - static_only)\n(mean +/- 95% CI)")
        axes[1].set_ylabel("Delta cosine")

    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--static-hook-mode",
        choices=["fusion_bypass", "zero_flow_tokens"],
        default="fusion_bypass",
        help="'fusion_bypass' sets z_fused = z_static directly; 'zero_flow_tokens' only zeroes flow_encoder output.",
    )
    parser.add_argument("--top-fraction", type=float, default=0.2, help="Fraction of masked tokens labeled dynamic")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=None, help="Bootstrap block size (default: ceil(n^(1/3)))")
    parser.add_argument("--chart-output", default=None, help="Output PNG chart path (default: next to the report)")


def evaluate(args: argparse.Namespace, device: torch.device):
    config, model = build_pipeline(args, device)
    loader, window_ids = make_eval_loader(args, config)

    logger.info("Calibration pass: mean-position baseline and flow partition")
    calibration = collect_calibration(model, loader, args, device)

    static_hooks: list[tuple[nn.Module, Any]] = [(model.flow_encoder, ZeroFlowHook())]
    if args.static_hook_mode == "fusion_bypass":
        static_hooks.append((model.fusion, StaticOnlyFusionHook()))
    variant_hooks = {
        "full": None,
        "static_only": static_hooks,
        "random_flow": [(model.flow_encoder, RandomFlowHook(base_seed=args.seed + 1000))],
    }
    logger.info("Variant pass: %s on identical masks", ", ".join(VARIANTS))
    raw = {
        name: evaluate_variant(model, loader, name, calibration, args, device, hooks)
        for name, hooks in variant_hooks.items()
    }

    summary = {
        name: {
            **{key: summarize_values(values) for key, values in result["scores"].items()},
            **{f"num_{key}_tokens": value for key, value in result["counts"].items()},
        }
        for name, result in raw.items()
    }

    bootstrap = dict(n_boot=args.n_boot, block_size=args.block_size, seed=args.seed)
    paired = {
        f"{key}_full_vs_static_only": paired_bootstrap_ci(
            raw["full"]["scores"][key], raw["static_only"]["scores"][key], **bootstrap
        )
        for key in ("cos_global", "cos_dynamic", "cos_static")
    }
    paired["cos_dynamic_full_vs_random_flow"] = paired_bootstrap_ci(
        raw["full"]["scores"]["cos_dynamic"], raw["random_flow"]["scores"]["cos_dynamic"], **bootstrap
    )

    chart_path = Path(args.chart_output) if args.chart_output else Path(args.output).with_suffix(".png")
    generate_chart(summary, paired, chart_path)

    metrics = {"variants": summary, "paired_delta_bootstrap_95ci": paired}
    metadata = {
        "num_windows": len(window_ids),
        "num_batches": len(calibration["mag_patches"]),
        "unit": "per-batch mean over masked tokens",
        "top_fraction": args.top_fraction,
        "static_hook_mode": args.static_hook_mode,
        "chart": str(chart_path),
        "notes": {
            "scope": "All cosine and delta_motion metrics use masked tokens only.",
            "cosine": "z_pred and z_target are channel-wise LayerNorm-ed in FP32 before cosine similarity.",
            "delta_motion": "||z_fused[mask] - z_static[mask]|| / (||z_static[mask]|| + eps).",
            "dynamic_tokens": "Top fraction of masked patch tokens by optical-flow magnitude.",
            "mean_position_baseline": "Per-position mean target embedding accumulated in the calibration pass.",
            "mask_control": "RNG seed and mask-sampler counter are reset before every pass.",
        },
    }
    return metrics, metadata


def main() -> None:
    run_evaluation(TASK, "Motion ablation: static vs. dynamic flow for ToMo-JEPA", add_arguments, evaluate)


if __name__ == "__main__":
    main()
