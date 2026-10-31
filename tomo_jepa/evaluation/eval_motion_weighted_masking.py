"""Compare random and motion-weighted masking for latent prediction.

Traffic-agent instances are detected with YOLOv8-seg. Optical flow is corrected
for background ego-motion estimated outside the instance masks; agents whose
mean residual motion exceeds the configured threshold are labeled dynamic.
Patch-level motion scores then drive either random masking or motion-weighted
masking. ToMo-JEPA predictions are compared with the Drive-JEPA predictor baseline
on the same masked tokens, separately for static and dynamic regions. The
evaluation reports cosine similarity, paired deltas, bootstrap confidence
intervals, and optional per-window logs and heatmaps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from ..perception import CausalPredictor
from ..perception.masking import MaskSampler
from ..pipeline import ToMoJEPAPipeline
from .common import ABLATIONS, autocast, build_pipeline, logger, make_eval_loader, run_evaluation
from .metrics import bootstrap_ci, token_cosine_similarity

TASK = "motion_weighted_masking"
PATCH_SIZE = 16
DYNAMIC_PATCH_RATIO = 0.25
# COCO: person, car, motorcycle, bus, truck
TRAFFIC_CLASSES = {0, 2, 3, 5, 7}
STRATEGIES = ("random_mask", "motion_weighted_mask")
REGIONS = ("static", "dynamic")


# ==========================================
# 1. INSTANCE MASK, EGO-MOTION & DYNAMIC REGIONS
# ==========================================

def extract_agent_instances(seg_model: YOLO, frames: torch.Tensor, device) -> list[torch.Tensor]:
    """Returns instance masks for each frame.

    frames: [T,3,H,W] in [0,1]; returns list of [K,H,W] bool tensors.
    """
    height, width = frames.shape[-2:]
    images = (frames.permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    results = seg_model(
        list(images),
        device=0 if device.type == "cuda" else "cpu",
        retina_masks=True,  # mask has the same size as the input image, not letterboxed
        verbose=False,
    )
    out = []
    for result in results:
        if result.masks is None:
            out.append(torch.zeros(0, height, width, dtype=torch.bool))
            continue
        classes = result.boxes.cls.cpu().numpy().astype(int)
        keep = [i for i, c in enumerate(classes) if c in TRAFFIC_CLASSES]
        masks = result.masks.data.cpu()[keep] > 0.5 if keep else torch.zeros(0, height, width)
        out.append(masks.bool())
    return out


def full_res_flow(estimator, images: torch.Tensor) -> torch.Tensor:
    """Flow [T-1,2,H,W] in pixels; estimator.forward only returns 1/8 resolution flow, so call backbone directly."""
    scale = 255.0 if images.max() <= 1.0 else 1.0
    flow_up, _ = estimator.backbone(images[0, :-1] * scale, images[0, 1:] * scale)
    return flow_up


def compute_relative_motion(flow: torch.Tensor, instances: torch.Tensor, motion_threshold: float):
    """v_rel = v_raw - v_bg; an agent is dynamic if mean |v_rel| over the instance > tau.

    flow: [2,H,W]; instances: [K,H,W] bool. Returns (|v_rel| [H,W], is_dynamic [H,W] bool).
    """
    instances = instances.to(flow.device)
    union = instances.any(dim=0) if len(instances) else torch.zeros_like(flow[0], dtype=torch.bool)
    background = ~union
    if background.any():
        v_bg = flow[:, background].median(dim=1).values.view(2, 1, 1)
    else:
        v_bg = torch.zeros(2, 1, 1, device=flow.device)
    rel_mag = torch.linalg.vector_norm(flow - v_bg, dim=0)

    is_dynamic = torch.zeros_like(union)
    for instance in instances:
        if rel_mag[instance].mean() > motion_threshold:
            is_dynamic |= instance
    return rel_mag, is_dynamic


def patch_level_labels(is_dynamic: torch.Tensor, rel_mag: torch.Tensor):
    """Dynamic labels and masking weights per patch (row-major), only considering motion of dynamic agents."""
    h, w = is_dynamic.shape
    shape = (h // PATCH_SIZE, PATCH_SIZE, w // PATCH_SIZE, PATCH_SIZE)
    ratio = is_dynamic.float().view(shape).mean(dim=(1, 3))
    weight = (rel_mag * is_dynamic.float()).view(shape).mean(dim=(1, 3))
    return (ratio > DYNAMIC_PATCH_RATIO).flatten(), weight.flatten()


# ==========================================
# 2. MASK SAMPLERS
# ==========================================

def _split(z_task: torch.Tensor, mask_indices: torch.Tensor):
    total, dim = z_task.shape[1], z_task.shape[2]
    mask_indices = mask_indices.to(device=z_task.device, dtype=torch.long).sort(dim=1).values
    context_indices = MaskSampler.complement_indices(mask_indices, total)
    z_masked = torch.gather(z_task, 1, context_indices.unsqueeze(-1).expand(-1, -1, dim))
    return z_masked, context_indices, mask_indices


class RandomMaskSampler(nn.Module):
    def __init__(self, num_masked: int, seed: int = 0) -> None:
        super().__init__()
        self.num_masked = num_masked
        self.generator = torch.Generator().manual_seed(seed)

    def forward(self, z_task: torch.Tensor):
        batch, total, _ = z_task.shape
        rows = [torch.randperm(total, generator=self.generator)[: self.num_masked] for _ in range(batch)]
        return _split(z_task, torch.stack(rows))


class MotionWeightedMaskSampler(nn.Module):
    def __init__(self, num_masked: int, seed: int = 0, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_masked = num_masked
        self.eps = eps
        self.generator = torch.Generator().manual_seed(seed)
        self.scores: torch.Tensor | None = None

    def set_scores(self, scores: torch.Tensor) -> None:
        self.scores = scores.detach().to(device="cpu", dtype=torch.float64)

    def forward(self, z_task: torch.Tensor):
        if self.scores is None:
            raise RuntimeError("Must call set_scores() before forward")
        weights = self.scores.clamp_min(0.0) + self.eps
        mask = torch.multinomial(weights, self.num_masked, replacement=False, generator=self.generator)
        return _split(z_task, mask)


def strategy_seeds(window_id: int, base_seed: int, legacy: bool = False) -> dict[str, int]:
    """Return reproducible, strategy-specific seeds, varying by window by default."""
    if legacy:
        return {"random_mask": 1, "motion_weighted_mask": 2}
    return {
        "random_mask": base_seed + 2 * window_id,
        "motion_weighted_mask": base_seed + 2 * window_id + 1,
    }


# ==========================================
# 3. HEATMAP
# ==========================================

def render_heatmap(frame, is_dynamic, patch_weight, mask_cos: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path):
    """Overlay dynamic agents and motion-weighted predictions on the source frame."""
    grid = frame.shape[0] // PATCH_SIZE
    fig = plt.figure(figsize=(12, 8))
    layout = fig.add_gridspec(2, 6)
    axes = (
        fig.add_subplot(layout[0, 0:2]),
        fig.add_subplot(layout[0, 2:4]),
        fig.add_subplot(layout[0, 4:6]),
        fig.add_subplot(layout[1, 1:3]),
        fig.add_subplot(layout[1, 3:5]),
    )
    extent = [0, frame.shape[1], frame.shape[0], 0]

    axes[0].imshow(frame)
    axes[0].set_title("Frame")
    axes[1].imshow(frame)
    axes[1].imshow(np.ma.masked_where(~is_dynamic, is_dynamic), alpha=0.6, cmap="autumn")
    axes[1].set_title("Dynamic agents (ego-motion removed)")
    axes[2].imshow(frame)
    im = axes[2].imshow(patch_weight.reshape(grid, grid), alpha=0.6, cmap="magma", extent=extent, interpolation="nearest")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Motion-weighted priority")

    def overlay(ax, indices, values, title, cmap, vmin, vmax):
        cells = np.full(grid * grid, np.nan)
        cells[indices] = values
        ax.imshow(frame)
        im = ax.imshow(
            np.ma.masked_invalid(cells.reshape(grid, grid)),
            alpha=0.7, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)

    indices, tomo, base = mask_cos["motion_weighted_mask"]
    overlay(axes[3], indices, tomo, "Motion-weighted: ToMo-JEPA cosine", "RdYlGn", 0.0, 1.0)
    overlay(axes[4], indices, base, "Motion-weighted: Drive-JEPA cosine", "RdYlGn", 0.0, 1.0)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ==========================================
# 4. EVALUATION
# ==========================================

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--motion-threshold", type=float, default=0.5, help="tau_motion (px/frame)")
    parser.add_argument("--seg-model", default="checkpoints/yolov8n-seg.pt",
                        help="YOLOv8-seg weights; downloaded here if missing")
    parser.add_argument(
        "--baseline-predictor-depth",
        type=int,
        default=0,
        help="Baseline predictor depth (0 = use ToMoJEPAConfig.predictor_depth)",
    )
    parser.add_argument("--num-heatmaps", type=int, default=5, help="Number of evenly spaced heatmaps to generate (0 = disabled)")
    parser.add_argument("--legacy-seed", action="store_true", help="Reuse fixed masking seeds 1/2 for every window")


def evaluate(args: argparse.Namespace, device: torch.device):
    if args.batch_size != 1:
        raise ValueError("this evaluation processes one window at a time; use --batch-size 1")
    if args.num_heatmaps < 0:
        raise ValueError("--num-heatmaps must be >= 0")

    config, pipeline = build_pipeline(args, device)
    # Baseline: original Drive-JEPA predictor (weights from vitl_checkpoint_path); shares the encoder with the pipeline.
    baseline_predictor = CausalPredictor(
        checkpoint_path=config.vitl_checkpoint_path,
        vjepa2_root=config.vjepa2_root,
        image_size=config.image_size,
        num_transitions=config.history_length - 1,
        latent_dim=config.latent_dim,
        confounder_dim=config.confounder_dim,
        predictor_embed_dim=config.predictor_embed_dim,
        predictor_depth=args.baseline_predictor_depth or config.predictor_depth,
        predictor_num_heads=config.predictor_num_heads,
        num_mask_tokens=config.predictor_num_mask_tokens,
        mode=config.predictor_confounder_mode,
    ).to(device).eval()
    Path(args.seg_model).expanduser().parent.mkdir(parents=True, exist_ok=True)
    seg_model = YOLO(args.seg_model)

    loader, window_ids = make_eval_loader(args, config)
    out_path = Path(args.output)
    log_path = out_path.with_suffix(".jsonl")
    heatmap_dir = out_path.parent / f"{out_path.stem}_heatmaps"
    if args.num_heatmaps > 0:
        heatmap_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_positions = (
        {int(i) for i in np.linspace(0, len(window_ids) - 1, min(args.num_heatmaps, len(window_ids))).round()}
        if args.num_heatmaps > 0 and window_ids
        else set()
    )

    new_group = lambda: {s: {r: [] for r in REGIONS} for s in STRATEGIES}
    tomo_scores, baseline_scores, delta_scores = new_group(), new_group(), new_group()
    # Mean predictor: cos(m, t) = (m . t_hat) / |m|, so only the mean of t_hat per region is stored.
    target_sum, target_count = None, 0
    mean_predictor_hat = {r: [] for r in REGIONS}

    with torch.no_grad(), log_path.open("w", encoding="utf-8") as log_file:
        for position, batch in enumerate(loader):
            window_id = window_ids[position]
            images = batch.images.to(device)
            batch.images = images
            _, frames, _, height, width = images.shape
            pairs = frames - 1
            tokens_per_transition = (height // PATCH_SIZE) * (width // PATCH_SIZE)
            num_masked = int(config.mask_ratio * pairs * tokens_per_transition)

            # Full-resolution flow [T-1,2,H,W] (px) and instance masks of the first frame of each pair
            flow = full_res_flow(pipeline.flow_estimator, images)
            instances = extract_agent_instances(seg_model, images[0, :-1], device)

            dyn_labels, weights, dyn_pixels = [], [], []
            for t in range(pairs):
                rel_mag, is_dynamic = compute_relative_motion(flow[t], instances[t], args.motion_threshold)
                labels, weight = patch_level_labels(is_dynamic, rel_mag)
                dyn_labels.append(labels)
                weights.append(weight)
                dyn_pixels.append(is_dynamic)
            patch_dynamic = torch.cat(dyn_labels).cpu()
            patch_weight = torch.cat(weights).cpu()
            del flow, instances
            if device.type == "cuda":
                torch.cuda.empty_cache()

            seeds = strategy_seeds(window_id, args.seed, args.legacy_seed)
            samplers = {
                "random_mask": RandomMaskSampler(num_masked, seed=seeds["random_mask"]),
                "motion_weighted_mask": MotionWeightedMaskSampler(num_masked, seed=seeds["motion_weighted_mask"]),
            }
            samplers["motion_weighted_mask"].set_scores(patch_weight.unsqueeze(0))

            heat = {}
            record = {"window_id": window_id, "mask_seeds": seeds}
            for name, sampler in samplers.items():
                pipeline.mask_sampler = sampler
                with autocast(args, device):
                    out = pipeline(batch)
                    # Same mask/context but only static tokens from the encoder, without motion/factorizer/confounder.
                    z_context = ToMoJEPAPipeline._gather_tokens(out.static_features.flatten(1, 2), out.context_indices)
                    base_pred = baseline_predictor(z_context, None, out.context_indices, out.mask_indices)[0]
                mask = out.mask_indices[0].cpu()
                z_target = out.target_patches[0].float()
                cosine = token_cosine_similarity(out.z_pred[0], z_target).cpu()
                base_cosine = token_cosine_similarity(base_pred, z_target).cpu()
                is_dyn = patch_dynamic[mask]

                record[name] = {}
                for region, sel in (("dynamic", is_dyn), ("static", ~is_dyn)):
                    tomo_mean = base_mean = None
                    if sel.any():
                        tomo_mean = cosine[sel].mean().item()
                        base_mean = base_cosine[sel].mean().item()
                        tomo_scores[name][region].append(tomo_mean)
                        baseline_scores[name][region].append(base_mean)
                        delta_scores[name][region].append(tomo_mean - base_mean)
                    record[name][f"{region}_cosine_similarity"] = tomo_mean
                    record[name][f"baseline_{region}_cosine_similarity"] = base_mean
                    record[name][f"num_{region}_masked"] = int(sel.sum())

                if name == "random_mask":
                    target_cpu = z_target.cpu()
                    target_hat = F.normalize(target_cpu, dim=-1)
                    target_sum = target_cpu.sum(0) if target_sum is None else target_sum + target_cpu.sum(0)
                    target_count += target_cpu.shape[0]
                    for region, sel in (("dynamic", is_dyn), ("static", ~is_dyn)):
                        mean_predictor_hat[region].append(target_hat[sel].mean(0) if sel.any() else None)

                first = mask < tokens_per_transition
                heat[name] = (mask[first].numpy(), cosine[first].numpy(), base_cosine[first].numpy())

            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()

            if position in heatmap_positions:
                frame = (images[0, 0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                render_heatmap(
                    frame,
                    dyn_pixels[0].cpu().numpy(),
                    patch_weight[:tokens_per_transition].numpy(),
                    heat,
                    heatmap_dir / f"heatmap_window_{window_id}.png",
                )
            logger.info("Window %d/%d (id=%d)", position + 1, len(window_ids), window_id)

    mean_vec = target_sum / target_count
    mean_predictor_scores = {
        region: [float((hat @ mean_vec) / mean_vec.norm()) for hat in hats if hat is not None]
        for region, hats in mean_predictor_hat.items()
    }

    summarize = lambda group: {s: {r: bootstrap_ci(v) for r, v in regions.items()} for s, regions in group.items()}
    metrics = {
        "tomo_jepa": summarize(tomo_scores),
        "baseline_vjepa2": summarize(baseline_scores),
        "paired_delta_tomo_minus_baseline": summarize(delta_scores),
        "mean_predictor": {region: bootstrap_ci(values) for region, values in mean_predictor_scores.items()},
    }
    metadata = {
        "num_windows": len(window_ids),
        "motion_threshold_px": args.motion_threshold,
        "resolution": f"{config.image_size[0]}x{config.image_size[1]}",
        "segmentation_model": args.seg_model,
        "mask_ratio": config.mask_ratio,
        "mean_predictor_tokens": "random_mask masked tokens",
        "baseline_predictor_checkpoint": config.vitl_checkpoint_path,
        "baseline_predictor_depth": args.baseline_predictor_depth or config.predictor_depth,
        "ablation": [name for name in ABLATIONS if getattr(config, f"ablation_{name}")],
        "seed_mode": "legacy_fixed_1_2" if args.legacy_seed else "per_window",
        "window_log": str(log_path),
    }
    return metrics, metadata


def main() -> None:
    run_evaluation(
        TASK,
        "Motion-weighted masking evaluation",
        add_arguments,
        evaluate,
        batch_size=1,
        amp="none",
    )


if __name__ == "__main__":
    main()
