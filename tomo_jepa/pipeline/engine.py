"""Training engine execution logic for ToMo-JEPA."""

from __future__ import annotations

import torch
from torch import nn

from ..contracts import FrameBatch, ModelOutput
from .pipeline import ToMoJEPAPipeline


def train_step(
    model: ToMoJEPAPipeline | nn.DataParallel | nn.parallel.DistributedDataParallel,
    batch: FrameBatch,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module | None = None,
) -> ModelOutput:
    """Run forward pass, compute total loss, backpropagate, step optimizer, and update target encoder via EMA."""
    base_model = getattr(model, "module", model)
    model.train()

    active_loss_fn = loss_fn if loss_fn is not None else base_model.loss_fn
    use_bfloat16 = (
        batch.images.device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    with torch.autocast(
        device_type=batch.images.device.type,
        dtype=torch.bfloat16,
        enabled=use_bfloat16,
    ):
        output = model(batch)
        losses = active_loss_fn(
            z_pred=output.z_pred,
            z_target=output.target_patches,
            z_task=output.z_task,
            z_exogenous=output.z_exogenous,
        )
    total_loss = losses["total"]
    output.losses = losses

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    base_model.update_target_encoder()

    return output