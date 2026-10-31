"""Typed tensor contracts shared across ToMo-JEPA stages."""

from dataclasses import dataclass

import torch


@dataclass
class FrameBatch:
    """A batch of temporally ordered ToMo image frames.

    ``images`` has shape ``[B, T, C, H, W]`` where ``B`` is batch size,
    ``T`` is sequence/history length, ``C`` is normally RGB=3, and ``H, W``
    are the resized image dimensions. The ToMo loader returns ``float32``
    images in the ``[0, 1]`` range.
    """

    images: torch.Tensor

    def validate(self) -> None:
        if self.images.ndim != 5:
            raise ValueError("images must have shape [batch, time, channels, height, width]")


@dataclass
class ModelOutput:
    """Canonical output for ToMoJEPAPipeline supporting training, evaluation, and ablation studies.

    Shapes:
    - ``z_pred``: [B, M, D] - Predicted latent representations
    - ``target_patches``: [B, M, D] - Target patch representations
    - ``static_features``: [B, T-1, N, D] - Raw static ViT features
    - ``fused_features``: [B, T-1, N, D] - Fused multimodal features
    - ``z_task``: [B, (T-1)*N, D] or [B, T-1, N, D] - Task-specific representations
    - ``dynamic_features``: Optional [B, T-1, N, D] - Motion flow tokens
    - ``z_exogenous``: Optional [B, (T-1)*N, D] - Exogenous features
    - ``P_task``: Optional [D, D_task] - Task projection matrix for do(a_t) interventions
    - ``P_exogenous``: Optional [D, D_exo] - Exogenous projection matrix
    - ``U``: Optional [B, U_dim] - Confounder context vector
    - ``mask_indices``: Optional [B, M] - Indices of masked target patches
    - ``context_indices``: Optional [B, N_ctx] - Indices of visible context patches
    - ``losses``: Optional dict of computed losses
    """

    # 1. Standard Prediction Outputs
    z_pred: torch.Tensor
    target_patches: torch.Tensor

    # 2. Intermediate Representations
    static_features: torch.Tensor
    fused_features: torch.Tensor
    z_task: torch.Tensor
    dynamic_features: torch.Tensor | None = None

    # 3. Disentanglement & Causal Outputs
    z_exogenous: torch.Tensor | None = None
    P_task: torch.Tensor | None = None
    P_exogenous: torch.Tensor | None = None
    U: torch.Tensor | None = None

    # 4. Masking & Diagnostic Tools
    mask_indices: torch.Tensor | None = None
    context_indices: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None
