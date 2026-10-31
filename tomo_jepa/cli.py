"""Command-line runner for ToMo-JEPA training."""

from __future__ import annotations

import atexit
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from .config import ToMoJEPAConfig
from .data import make_tomo_dataloader
from .evaluation import HardwareTelemetryMonitor
from .pipeline import ToMoJEPAPipeline, save_checkpoint, load_checkpoint, train_step


def setup_vjepa2_path(vjepa2_root: str) -> None:
    """Add the V-JEPA2 root directory to sys.path for loading pretrained checkpoints."""
    path = Path(vjepa2_root).expanduser()

    if not path.is_absolute():
        workspace_root = Path(__file__).resolve().parents[1]
        path = (workspace_root / path).resolve()

    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def get_scenario_info(config: ToMoJEPAConfig) -> tuple[str, dict[str, bool]]:
    """Determine the active Ablation Study scenario name and config breakdown."""
    ablation_flags = {
        "motion_branch": config.ablation_motion_branch,
        "confounder": config.ablation_confounder,
        "factorizer": config.ablation_factorizer,
    }
    scenario_name = "Full_ToMo_JEPA" if not any(ablation_flags.values()) else f"Ablation_{'_'.join([k for k, v in ablation_flags.items() if v])}"
    return scenario_name, ablation_flags


def _gradient_diagnostics(
    model: ToMoJEPAPipeline,
) -> dict[str, dict[str, float | int]]:
    """Return gradient norms for every top-level model component with gradients."""
    component_gradients: dict[str, list[torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        component = name.partition(".")[0] or "<root>"
        component_gradients.setdefault(component, []).append(
            parameter.grad.detach().float().norm()
        )

    return {
        component: {
            "norm": torch.linalg.vector_norm(torch.stack(norms)).item(),
            "parameters_with_grad": len(norms),
        }
        for component, norms in component_gradients.items()
    }


class _Tee:
    """Duplicate a text stream into a log file."""

    def __init__(self, stream, log_file) -> None:
        self.stream = stream
        self.log_file = log_file

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    if is_distributed:
        if local_rank < 0:
            raise RuntimeError("Distributed training requires LOCAL_RANK; launch with torchrun.")
        if torch.cuda.is_available() and local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA device(s) are visible. "
                f"Use --nproc_per_node={torch.cuda.device_count()} or expose more GPUs "
                "(check CUDA_VISIBLE_DEVICES / container --gpus)."
            )
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))

    if torch.cuda.is_available():
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    is_main = rank == 0
    torch.backends.cudnn.benchmark = device.type == "cuda"

    config = ToMoJEPAConfig()

    # Setup V-JEPA2 root path
    setup_vjepa2_path(config.vjepa2_root)

    # Determine the active Ablation Study scenario
    scenario_name, ablation_flags = get_scenario_info(config)
    log_dir = Path(config.output_log_dir)
    log_file_path = log_dir / f"metrics_{scenario_name}.jsonl"
    if is_main:
        log_dir.mkdir(parents=True, exist_ok=True)
        console_log = open(log_dir / f"train_{scenario_name}.log", "a", encoding="utf-8")
        console_log.write(f"\n===== Run started {datetime.now().isoformat()} =====\n")
        sys.stdout = _Tee(sys.stdout, console_log)
        sys.stderr = _Tee(sys.stderr, console_log)
        atexit.register(console_log.close)

    latest_ckpt_path = Path(config.output_checkpoint_path).expanduser()
    checkpoint_dir = latest_ckpt_path.parent
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = checkpoint_dir / f"best_{latest_ckpt_path.name}"

    if is_main:
        print("=" * 70)
        print(f"[INFO] Running Scenario : {scenario_name}")
        print(f"[INFO] Ablation Flags : {ablation_flags}")
        print(f"[INFO] Training Mode : {'DDP across ' + str(world_size) + ' processes' if is_distributed else 'Single process'}")
        print(f"[INFO] Logging metrics to: {log_file_path}")
        print(f"[INFO] Latest Checkpoint : {latest_ckpt_path}")
        print(f"[INFO] Best Checkpoint   : {best_ckpt_path}")
        print("=" * 70)

    # Each DDP process receives a distinct shard of the training dataset.
    dataloader = make_tomo_dataloader(config, distributed=is_distributed)
    sampler = dataloader.sampler if isinstance(dataloader.sampler, DistributedSampler) else None
    if is_main:
        print(f"[INFO] Using device: {device}")
        if device.type == "cuda":
            print(f"[INFO] GPU Device: {torch.cuda.get_device_name(device)}")

    # Pipeline Model
    model = ToMoJEPAPipeline(config)
    model = model.to(device)
    if is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    base_model = model.module if isinstance(model, DistributedDataParallel) else model

    # Optimizer Parameter Grouping
    decay_params = []
    no_decay_params = []

    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        # Apply no weight decay to bias and LayerNorm/BatchNorm
        if "bias" in name or "norm" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": getattr(config, "weight_decay", 0.01)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    # Resume from Checkpoint
    start_epoch = 1
    best_loss = float("inf")

    resume_path = (
        Path(config.resume_checkpoint_path).expanduser()
        if config.resume_checkpoint_path
        else None
    )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
    if resume_path is None and latest_ckpt_path.exists() and not config.pretrained:
        resume_path = latest_ckpt_path

    if resume_path is not None:
        if is_main:
            print(f"[INFO] Resuming from checkpoint: {resume_path}")
        ckpt_data, epoch, optimizer = load_checkpoint(
            checkpoint_path=resume_path,
            model=base_model,
            optimizer=optimizer,
            strict=False,
        )
        if epoch is not None:
            if is_main:
                print(f"[INFO] Resuming training from epoch {epoch + 1}.")
            start_epoch = epoch + 1
            if "best_loss" in ckpt_data:
                best_loss = float(ckpt_data["best_loss"])
    else:
        if is_main:
            print("[INFO] No resume checkpoint configured. Starting fresh training.")

    num_epochs = config.num_epochs
    n_steps_per_epoch = config.n_steps_per_epoch
    total_steps = len(dataloader)
    telemetry_monitor = None
    if is_main:
        telemetry_log_path = log_dir / f"hardware_{scenario_name}.jsonl"
        telemetry_monitor = HardwareTelemetryMonitor(telemetry_log_path)
        telemetry_monitor.start()
        atexit.register(telemetry_monitor.stop)
        print(f"[INFO] Background hardware telemetry: {telemetry_log_path}")

    for epoch in range(start_epoch, num_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running_epoch_loss = 0.0
        num_batches_in_epoch = 0

        for batch_idx, batch in enumerate(dataloader, start=1):
            step_start_time = time.time()
            # Execute full train step (Forward -> Loss -> Backprop -> Optimizer step -> EMA update)
            batch.images = batch.images.to(device)
            output = train_step(model, batch, optimizer, base_model.loss_fn)
            step_time = time.time() - step_start_time

            current_losses = {
                k: float(v.detach().cpu()) for k, v in output.losses.items()
            }
            step_loss = current_losses.get("total", 0.0)
            running_epoch_loss += step_loss
            num_batches_in_epoch += 1

            if batch_idx % n_steps_per_epoch == 0 or batch_idx == total_steps:
                if is_distributed:
                    loss_values = torch.tensor(
                        [current_losses[name] for name in current_losses],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.all_reduce(loss_values, op=dist.ReduceOp.SUM)
                    loss_values /= world_size
                    current_losses = dict(zip(current_losses, loss_values.cpu().tolist()))
                    step_loss = current_losses.get("total", 0.0)

                if is_main:
                    gradient_norms = _gradient_diagnostics(base_model)
                    gradient_summary = " | ".join(
                        f"{component}: {metrics['norm']:.4e} "
                        f"({metrics['parameters_with_grad']} params)"
                        for component, metrics in gradient_norms.items()
                    ) or "none"
                    loss_str = " | ".join(
                        [f"{k}: {v:.4f}" for k, v in current_losses.items()]
                    )
                    global_batch_size = config.batch_size * world_size
                    hardware_metrics = {
                        "samples_per_sec": (
                            global_batch_size / step_time if step_time > 0 else 0.0
                        ),
                        "peak_allocated_memory_gb": 0.0,
                        "peak_reserved_memory_gb": 0.0,
                    }
                    if device.type == "cuda":
                        hardware_metrics.update(
                            {
                                "peak_allocated_memory_gb": (
                                    torch.cuda.max_memory_allocated(device) / (1024**3)
                                ),
                                "peak_reserved_memory_gb": (
                                    torch.cuda.max_memory_reserved(device) / (1024**3)
                                ),
                            }
                        )
                    print(
                        f"[Epoch {epoch:03d}/{num_epochs:03d}][Step {batch_idx:04d}/{total_steps:04d}] "
                        f"Loss: {step_loss:.4f} ({loss_str}) | "
                        f"Speed: {step_time:.2f}s/step\n"
                        f"      -> Gradient Norms by component | {gradient_summary}\n"
                        f"      -> Hardware | Throughput: {hardware_metrics['samples_per_sec']:.2f} samples/s | "
                        f"Peak VRAM: {hardware_metrics['peak_allocated_memory_gb']:.2f} GB allocated / "
                        f"{hardware_metrics['peak_reserved_memory_gb']:.2f} GB reserved"
                    )
                    step_log_record = {
                        "timestamp": datetime.now().isoformat(),
                        "epoch": epoch,
                        "step": batch_idx,
                        "total_steps": total_steps,
                        "scenario": scenario_name,
                        "losses": current_losses,
                        "grad_norms": gradient_norms,
                        "hardware": hardware_metrics,
                        "step_time_sec": round(step_time, 3),
                    }
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(step_log_record) + "\n")

                    # Checkpoints are written only by rank zero.
                    step_ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}_step_{batch_idx:06d}.pt"
                    saved_path = save_checkpoint(
                        checkpoint_path=step_ckpt_path,
                        model=base_model,
                        epoch=epoch,
                        optimizer=optimizer,
                    )
                    print(f"[INFO] Mid-epoch checkpoint saved to {saved_path}")

        # Compute the average loss for the epoch
        if is_distributed:
            epoch_stats = torch.tensor(
                [running_epoch_loss, num_batches_in_epoch],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)
            running_epoch_loss, num_batches_in_epoch = epoch_stats.tolist()
        epoch_avg_loss = running_epoch_loss / max(1, num_batches_in_epoch)

        if is_main:
            saved_path = save_checkpoint(
                checkpoint_path=latest_ckpt_path,
                model=base_model,
                epoch=epoch,
                optimizer=optimizer,
            )
            print(
                f"[INFO] [Epoch {epoch:03d}/{num_epochs:03d}] "
                f"Average loss: {epoch_avg_loss:.4f} over {int(num_batches_in_epoch)} steps"
            )
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "epoch": epoch,
                    "scenario": scenario_name,
                    "epoch_avg_loss": epoch_avg_loss,
                }) + "\n")
            print(f"[INFO] [Epoch {epoch:03d}] Latest checkpoint updated at {saved_path}")

            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                saved_path = save_checkpoint(
                    checkpoint_path=best_ckpt_path,
                    model=base_model,
                    epoch=epoch,
                    optimizer=optimizer,
                )
                print(f"[INFO] [Epoch {epoch:03d}] New best loss ({best_loss:.4f})! Saved best checkpoint to {saved_path}")

    if is_main:
        print({
            "status": "Train step completed successfully",
            "latest_checkpoint": str(latest_ckpt_path),
            "best_checkpoint": str(best_ckpt_path),
            "best_loss": best_loss,
            "static_features_shape": tuple(output.static_features.shape),
            "target_patches_shape": tuple(output.target_patches.shape),
            "dynamic_features_shape": tuple(output.dynamic_features.shape) if output.dynamic_features is not None else None,
            "fused_features_shape": tuple(output.fused_features.shape) if output.fused_features is not None else None,
            "confounder_features_shape": tuple(output.U.shape) if output.U is not None else None,
            "z_task_shape": tuple(output.z_task.shape) if output.z_task is not None else None,
            "z_exogenous_shape": tuple(output.z_exogenous.shape) if output.z_exogenous is not None else None,
            "P_task_shape": tuple(output.P_task.shape) if output.P_task is not None else None,
            "P_exogenous_shape": tuple(output.P_exogenous.shape) if output.P_exogenous is not None else None,
            "jepa_predicted_features_shape": tuple(output.z_pred.shape),
            "final_losses": {name: float(val.detach()) for name, val in output.losses.items()},
        })

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
    if telemetry_monitor is not None:
        telemetry_monitor.stop()
        atexit.unregister(telemetry_monitor.stop)


if __name__ == "__main__":
    main()