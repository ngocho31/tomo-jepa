"""Evaluate ego-vehicle trajectory planning with ADE and FDE.

The ToMo-JEPA backbone is frozen and a lightweight MLP trajectory head is
trained on pooled task latents and the confounder vector. It predicts future
positions in the ego frame at the last observed frame (x forward, y left).
Ground-truth positions and orientation are read from ``can_bus`` for frames
immediately following the observation window.

Each sample contains ``history_length + horizon`` frames: the observation
history is passed to the model, while the future horizon is used only for
ground truth. The script trains or loads separate ToMo-JEPA and static V-JEPA2
trajectory heads, then compares ADE/FDE with bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ToMoJEPAConfig
from ..data import ToMoEpisodeDataset, collate_tomo_samples, future_trajectory_ego_frame
from .common import autocast, build_pipeline, logger, make_eval_loader, run_evaluation
from .metrics import bootstrap_ci, compute_ade_fde, paired_bootstrap_ci

TASK = "ego_trajectory"


class TrajectoryDecoderHead(nn.Module):
    """MLP mapping pooled z_task [B, D] and U_t [B, C] to a trajectory [B, T, 2]."""

    def __init__(self, latent_dim: int = 1024, context_dim: int = 128, pred_horizon: int = 30, hidden_dim: int = 256):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.fc = nn.Sequential(
            nn.LayerNorm(latent_dim + context_dim),
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_horizon * 2),
        )

    def forward(self, z_pooled: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_pooled, u_t], dim=-1)).view(-1, self.pred_horizon, 2)


def collate_with_can_bus(samples):
    return collate_tomo_samples(samples), torch.stack([sample["can_bus"] for sample in samples])


def add_arguments(parser: argparse.ArgumentParser) -> None:
    default_config = ToMoJEPAConfig()
    parser.add_argument("--train-split", default=default_config.dataset_split, help="Split used to train the heads")
    parser.add_argument(
        "--traj-head-ckpt",
        default="checkpoints/trajectory_head.pt",
        help="ToMo-JEPA trajectory-head checkpoint; trained and saved if it does not exist",
    )
    parser.add_argument(
        "--baseline-head-ckpt",
        default="checkpoints/trajectory_head_vjepa2.pt",
        help="Baseline V-JEPA2 trajectory-head checkpoint; independent of the ToMo-JEPA checkpoint",
    )
    parser.add_argument("--horizon", type=int, default=40, help="Number of future steps (10 Hz: 40 steps = 4 s)")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=5, help="Step between dataset windows")
    parser.add_argument("--retrain", action="store_true", help="Retrain the ToMo-JEPA head even if a checkpoint exists")
    parser.add_argument("--retrain-baseline", action="store_true", help="Retrain the baseline head even if a checkpoint exists")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=2, help="Transitions processed per chunk to reduce peak VRAM")
    parser.add_argument("--flow-on-cpu", action="store_true", help="Run FlowFormer++ on CPU to reduce GPU memory use")


def make_loader(args, config, split: str):
    dataset = ToMoEpisodeDataset(
        args.data_root,
        split=split,
        history_length=config.history_length,
        future_length=args.horizon,
        stride=args.stride,
        image_size=config.image_size,
        max_cached_episodes=config.max_cached_episodes,
    )
    logger.info("Split '%s': %d windows", split, len(dataset))
    if split == args.train_split and split != args.split:
        # Window selection options only restrict the evaluation split, not head training.
        args = argparse.Namespace(**{**vars(args), "start_window": 0, "end_window": None, "window_step": 1, "max_windows": None})
    return make_eval_loader(args, config, dataset=dataset, collate_fn=collate_with_can_bus)[0]


def window_features(pipeline, images: torch.Tensor, args, device):
    """Pooled (z_task [B,D], U [B,C] or None, z_static [B,D]) computed ``--chunk`` transitions at a time to limit VRAM.

    ``z_static`` is the frozen V-JEPA2 baseline feature.
    """
    task_sum, static_sum, fused_pooled = 0.0, 0.0, []
    total = images.shape[1] - 1
    for start in range(0, total, args.chunk):
        with autocast(args, device):
            encoded = pipeline.encode_transitions(images[:, start:start + args.chunk + 1])
        task_sum = task_sum + encoded["z_task"].float().mean(dim=2).sum(dim=1)
        static_sum = static_sum + encoded["static"].float().mean(dim=2).sum(dim=1)
        fused_pooled.append(encoded["fused"].float().mean(dim=2))
    u = pipeline.summarize_context(torch.cat(fused_pooled, dim=1))
    return task_sum / total, u, static_sum / total


@torch.no_grad()
def extract_features(pipeline, loader, args, config, device):
    """Returns (z_pooled [N,D], u_t [N,C], z_static [N,D], gt_traj [N,H,2]) on CPU."""
    zs, us, statics, gts = [], [], [], []
    for position, (batch, can_bus) in enumerate(loader):
        z, u, z_static = window_features(pipeline, batch.images.to(device), args, device)
        zs.append(z.cpu())
        statics.append(z_static.cpu())
        us.append(u.float().cpu() if u is not None else torch.zeros(len(can_bus), config.confounder_dim))
        gts.append(future_trajectory_ego_frame(can_bus, config.history_length))
        if (position + 1) % 20 == 0 or position + 1 == len(loader):
            logger.info("Extracted features for batch %d/%d", position + 1, len(loader))
    return torch.cat(zs), torch.cat(us), torch.cat(statics), torch.cat(gts)


def train_head(head, z, u, gt, device, args):
    head.to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    count = len(z)
    for epoch in range(args.epochs):
        order = torch.randperm(count)
        total = 0.0
        for i in range(0, count, args.head_batch_size):
            idx = order[i:i + args.head_batch_size]
            loss = F.smooth_l1_loss(head(z[idx].to(device), u[idx].to(device)), gt[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("Head epoch %d/%d loss=%.4f", epoch + 1, args.epochs, total / count)
    return head.eval()


@torch.no_grad()
def predict(head, z, u, device, batch_size: int = 1024) -> torch.Tensor:
    head.eval()
    return torch.cat([
        head(z[i:i + batch_size].to(device), u[i:i + batch_size].to(device)).cpu()
        for i in range(0, len(z), batch_size)
    ])


def evaluate(args: argparse.Namespace, device: torch.device):
    config, pipeline = build_pipeline(args, device, lightweight=True)
    if args.flow_on_cpu:
        pipeline.flow_estimator.to("cpu")

    head_path = Path(args.traj_head_ckpt)
    baseline_path = Path(args.baseline_head_ckpt)
    train_cache: list[torch.Tensor] = []

    def get_head(path: Path, context_dim: int, select_inputs, retrain: bool) -> TrajectoryDecoderHead:
        head = TrajectoryDecoderHead(config.latent_dim, context_dim, args.horizon)
        if path.is_file() and not retrain:
            saved = torch.load(path, map_location="cpu")
            if saved["pred_horizon"] != args.horizon:
                raise ValueError(f"head ckpt horizon={saved['pred_horizon']} != --horizon {args.horizon}")
            head.load_state_dict(saved["state_dict"])
            logger.info("Loaded trajectory head: %s", path)
            return head.to(device).eval()
        if not train_cache:
            train_cache.extend(extract_features(pipeline, make_loader(args, config, args.train_split), args, config, device))
        z, u, z_static, gt = train_cache
        head = train_head(head, *select_inputs(z, u, z_static), gt, device, args)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "pred_horizon": args.horizon}, path)
        logger.info("Saved trajectory head: %s", path)
        return head

    # Baseline V-JEPA2: static tokens of the frozen encoder only, with a separate head of the same architecture.
    tomo_inputs = lambda z, u, s: (z, u)
    base_inputs = lambda z, u, s: (s, s.new_zeros(len(s), 0))
    tomo_head = get_head(head_path, config.confounder_dim, tomo_inputs, args.retrain)
    base_head = get_head(baseline_path, 0, base_inputs, args.retrain_baseline)
    train_cache.clear()

    z, u, z_static, gt = extract_features(pipeline, make_loader(args, config, args.split), args, config, device)
    tomo_metrics, tomo_per_window = compute_ade_fde(predict(tomo_head, z, u, device), gt, args.fps)
    base_pred = predict(base_head, z_static, z_static.new_zeros(len(z_static), 0), device)
    base_metrics, base_per_window = compute_ade_fde(base_pred, gt, args.fps)

    metrics = {
        "tomo_jepa": {**tomo_metrics, "ADE_ci_95": bootstrap_ci(tomo_per_window)},
        "baseline_vjepa2": {**base_metrics, "ADE_ci_95": bootstrap_ci(base_per_window)},
        "delta_tomo_minus_baseline": {
            "metrics": {key: tomo_metrics[key] - base_metrics[key] for key in tomo_metrics},
            "ADE_ci_95": paired_bootstrap_ci(tomo_per_window, base_per_window),
        },
    }
    metadata = {
        "num_windows": int(len(gt)),
        "train_split": args.train_split,
        "split": args.split,
        "horizon_steps": args.horizon,
        "fps": args.fps,
        "obs_frames": config.history_length,
        "frame": "ego frame at last observed frame (x forward, y left), meters",
        "traj_head_ckpt": str(head_path),
        "baseline_head_ckpt": str(baseline_path),
        "baseline_encoder_checkpoint": config.vitl_checkpoint_path,
    }
    return metrics, metadata


def main() -> None:
    run_evaluation(
        TASK,
        "ToMo-JEPA ego-trajectory planning evaluation (ADE/FDE)",
        add_arguments,
        evaluate,
    )


if __name__ == "__main__":
    main()
