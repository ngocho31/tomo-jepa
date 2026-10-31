"""Shared harness for the evaluation scripts: arguments, model/data setup and report writing.

Every script exposes ``add_arguments(parser)`` for its task-specific options and
``evaluate(args, device) -> (metrics, metadata)``; ``run_evaluation`` wires them to
the common CLI and writes ``{"metadata": ..., "metrics": ...}`` to ``--output``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import logging
import random
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..cli import setup_vjepa2_path
from ..config import ToMoJEPAConfig
from ..data import ToMoEpisodeDataset, collate_tomo_samples
from ..pipeline import ToMoJEPAPipeline, load_checkpoint

ABLATIONS = ("motion_branch", "confounder", "factorizer")

logger = logging.getLogger("tomo_jepa.evaluation")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_common_arguments(
    parser: argparse.ArgumentParser,
    task: str,
    *,
    batch_size: int | None = None,
    amp: str = "fp16",
) -> None:
    default_config = ToMoJEPAConfig()
    evaluation_dir = Path(default_config.output_log_dir).parent / "evaluation"
    parser.add_argument("--checkpoint", default=default_config.output_checkpoint_path)
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        choices=ABLATIONS,
        help="Disable the selected branch(es) to match the checkpoint's training configuration",
    )
    parser.add_argument("--data-root", default=default_config.dataset_root)
    parser.add_argument("--split", default="test", help="Evaluation split (default: test)")
    parser.add_argument("--batch-size", type=int, default=batch_size or default_config.batch_size)
    parser.add_argument("--num-workers", type=int, default=default_config.num_workers)
    parser.add_argument("--start-window", type=int, default=0, help="First window index (inclusive)")
    parser.add_argument("--end-window", type=int, default=0, help="Last window index (exclusive); 0 = until the end")
    parser.add_argument("--window-step", type=int, default=1, help="Evaluate every N-th window in [start, end)")
    parser.add_argument("--max-windows", type=int, default=0, help="0 = no limit; otherwise keep the first N selected windows")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default=amp)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(evaluation_dir / f"{task}.json"), help="JSON report path")


def setup_runtime(args: argparse.Namespace) -> torch.device:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Initialize the cuSOLVER handle early (LatentFactorizer uses QR) while the GPU has free memory.
        torch.linalg.qr(torch.randn(8, 8, device=device))
    return device


def autocast(args: argparse.Namespace, device: torch.device):
    if args.amp == "none" or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if args.amp == "bf16" else torch.float16)


def build_pipeline(
    args: argparse.Namespace,
    device: torch.device,
    *,
    lightweight: bool = False,
) -> tuple[ToMoJEPAConfig, ToMoJEPAPipeline]:
    """Build the pipeline for ``args`` and load ``args.checkpoint``.

    ``lightweight`` drops the target encoder and predictor for scripts that only need the frozen features.
    """
    config = ToMoJEPAConfig(
        dataset_root=args.data_root,
        dataset_split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **{f"ablation_{name}": True for name in args.ablation},
    )
    setup_vjepa2_path(config.vjepa2_root)
    pipeline = ToMoJEPAPipeline(config).eval()
    load_checkpoint(args.checkpoint, pipeline, strict=True)
    if lightweight:
        del pipeline.target_encoder, pipeline.predictor
    pipeline.requires_grad_(False)
    return config, pipeline.to(device)


def select_window_ids(args: argparse.Namespace, total: int) -> list[int]:
    if args.window_step < 1:
        raise ValueError("--window-step must be >= 1")
    start, end = args.start_window, args.end_window or total
    if not 0 <= start < end <= total:
        raise ValueError(f"invalid window range [{start}, {end}) for dataset of {total} windows")
    window_ids = list(range(start, end, args.window_step))
    return window_ids[: args.max_windows] if args.max_windows else window_ids


def make_eval_loader(
    args: argparse.Namespace,
    config: ToMoJEPAConfig,
    *,
    dataset: Dataset | None = None,
    collate_fn: Callable = collate_tomo_samples,
) -> tuple[DataLoader, list[int]]:
    """Deterministic loader over the windows selected by the common ``--*-window`` options."""
    if dataset is None:
        dataset = ToMoEpisodeDataset(
            config.dataset_root,
            split=config.dataset_split,
            history_length=config.history_length,
            stride=config.stride,
            image_size=config.image_size,
            max_cached_episodes=config.max_cached_episodes,
        )
    window_ids = select_window_ids(args, len(dataset))
    logger.info("Evaluating %d windows", len(window_ids))
    loader = DataLoader(
        Subset(dataset, window_ids),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    return loader, window_ids


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=4, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Saved report: %s", path)
    return path


def run_evaluation(
    task: str,
    description: str,
    add_arguments: Callable[[argparse.ArgumentParser], None],
    evaluate: Callable[[argparse.Namespace, torch.device], tuple[dict[str, Any], dict[str, Any]]],
    **common_defaults: Any,
) -> dict[str, Any]:
    """Parse arguments, run ``evaluate`` and write the standardized report."""
    parser = argparse.ArgumentParser(description=description)
    add_common_arguments(parser, task, **common_defaults)
    add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = setup_runtime(args)
    logger.info("Task: %s | device: %s", task, device)
    metrics, extra_metadata = evaluate(args, device)

    report = {
        "metadata": {
            "task": task,
            "timestamp": datetime.datetime.now().isoformat(),
            "git_commit": _git_commit(),
            "device": str(device),
            "arguments": {key: _json_safe(value) for key, value in vars(args).items()},
            **extra_metadata,
        },
        "metrics": metrics,
    }
    save_report(report, args.output)
    print(json.dumps(report["metrics"], indent=4, default=str))
    return report
