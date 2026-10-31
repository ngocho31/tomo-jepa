"""ToMo-JEPA pipeline: Static + Motion + Fusion + Factorization + Confounder + Predictor."""

from __future__ import annotations

import torch
from torch import nn

from ..causal import ConfounderGRU, LatentFactorizer
from ..config import ToMoJEPAConfig
from ..contracts import FrameBatch, ModelOutput
from ..evaluation import ToMoJEPALoss
from ..perception import (
    CausalPredictor,
    MaskSampler,
    GatedCrossAttentionFusion,
    build_vit_encoders,
    build_motion_encoders,
)


class ToMoJEPAPipeline(nn.Module):
    """Runs each stage explicitly while keeping stages replaceable."""

    def __init__(self, config: ToMoJEPAConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Static Branch (Load ViT-L weights, freeze)
        self.context_encoder, self.target_encoder = build_vit_encoders(
            checkpoint_path=config.vitl_checkpoint_path,
            vjepa2_root=config.vjepa2_root,
            image_size=config.image_size,
        )

        # 2. Motion Branch (Load FlowFormer++ weights and create motion encoder)
        self.flow_estimator, self.flow_encoder = build_motion_encoders(
            flowformer_checkpoint_path=config.motion_estimator_checkpoint_path,
            flowformer_root=config.flowformer_root,
            token_dim=config.latent_dim,
            flow_channels=config.flow_channels,
        )

        # 4. Predictor
        self.predictor = CausalPredictor(
            checkpoint_path=config.vitl_checkpoint_path,
            vjepa2_root=config.vjepa2_root,
            image_size=config.image_size,
            num_transitions=config.history_length - 1,
            latent_dim=config.latent_dim,
            confounder_dim=config.confounder_dim,
            predictor_embed_dim=config.predictor_embed_dim,
            predictor_depth=config.predictor_depth,
            predictor_num_heads=config.predictor_num_heads,
            num_mask_tokens=config.predictor_num_mask_tokens,
            mode=config.predictor_confounder_mode,
        )

        # 5. Fusion, Factorization, Confounder
        self.fusion = GatedCrossAttentionFusion(
            config.latent_dim,
            num_heads=config.fusion_num_heads,
            fusion_scale=config.fusion_scale,
        )
        self.factorizer = LatentFactorizer(config.latent_dim, config.task_dim, enforce_orthogonality=config.enforce_orthogonality)
        self.confounder = ConfounderGRU(config.latent_dim, config.confounder_dim)

        # 6. Mask Sampler and Loss Function
        self.mask_sampler = MaskSampler(
            config.mask_ratio,
            image_size=config.image_size,
            vjepa2_root=config.vjepa2_root,
        )
        self.loss_fn = ToMoJEPALoss(
            lambda_jepa=config.jepa_loss_weight,
            lambda_orth=config.orthogonality_loss_weight,
        )

        # Configure which modules are trainable based on the config
        self.configure_trainable_modules()

    @staticmethod
    def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        """EMA-update target encoder after an optimizer step on the context branch."""
        momentum_value = self.config.target_ema_momentum if momentum is None else momentum
        for target, source in zip(
            self.target_encoder.backbone.parameters(),
            self.context_encoder.backbone.parameters(),
        ):
            target.mul_(momentum_value).add_(source, alpha=1.0 - momentum_value)

    def configure_trainable_modules(self) -> None:
        self.context_encoder.requires_grad_(not self.config.freeze_context_encoder)
        self.target_encoder.requires_grad_(not self.config.freeze_target_encoder)

        # Modules bypassed in forward() by an ablation receive no gradient; freezing them
        # keeps them out of the optimizer and avoids DDP unused-parameter errors.
        motion_active = not self.config.ablation_motion_branch
        self.flow_estimator.requires_grad_(motion_active and not self.config.freeze_flow_estimator)
        self.flow_encoder.requires_grad_(motion_active and not self.config.freeze_motion_encoder)
        self.fusion.requires_grad_(motion_active and not self.config.freeze_fusion)
        self.factorizer.requires_grad_(
            not self.config.ablation_factorizer and not self.config.freeze_factorizer
        )
        self.confounder.requires_grad_(
            not self.config.ablation_confounder and not self.config.freeze_confounder
        )
        self.predictor.requires_grad_(not self.config.freeze_predictor)

    def train(self, mode: bool = True) -> ToMoJEPAPipeline:
        """Set the module in training mode while respecting frozen components."""
        super().train(mode)
        if mode:
            if getattr(self.config, "freeze_context_encoder", True):
                self.context_encoder.eval()
            if getattr(self.config, "freeze_target_encoder", True):
                self.target_encoder.eval()
            if getattr(self.config, "freeze_flow_estimator", True):
                self.flow_estimator.eval()
            if getattr(self.config, "freeze_motion_encoder", True):
                self.flow_encoder.eval()
            if getattr(self.config, "freeze_fusion", True):
                self.fusion.eval()
            if getattr(self.config, "freeze_factorizer", True):
                self.factorizer.eval()
            if getattr(self.config, "freeze_confounder", True):
                self.confounder.eval()
            if getattr(self.config, "freeze_predictor", True):
                self.predictor.eval()
        return self

    def encode_transitions(
        self, images: torch.Tensor, context_tokens: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        """Static, motion, fusion and factorization stages (steps 1-4) for ``images`` [B, T, 3, H, W].

        All token tensors are [B, T-1, N_p, D]. ``context_tokens`` may be supplied to skip the static encoder.
        """
        # Step 1: Static branch — keep full patch tokens
        if context_tokens is None:
            context_tokens = self.context_encoder.forward_tokens(images)

        # Step 2: Motion branch
        if self.config.ablation_motion_branch:
            dynamic = None
        else:
            flow_device = next(self.flow_estimator.parameters()).device
            flow_maps = self.flow_estimator(images.to(flow_device)).to(images.device)
            dynamic = self.flow_encoder(flow_maps)

        # Step 3: Residual fusion (token-wise)
        fused = context_tokens if dynamic is None else self.fusion(context_tokens, dynamic)

        # Step 4: Latent factorization (token-wise)
        if self.config.ablation_factorizer:
            z_task, z_exogenous, P_task, P_exogenous = fused, None, None, None
        else:
            z_task, z_exogenous, P_task, P_exogenous = self.factorizer(fused)

        return {
            "static": context_tokens,
            "dynamic": dynamic,
            "fused": fused,
            "z_task": z_task,
            "z_exogenous": z_exogenous,
            "P_task": P_task,
            "P_exogenous": P_exogenous,
        }

    def summarize_context(self, fused_global: torch.Tensor) -> torch.Tensor | None:
        """Step 5: confounder vector U from patch-pooled fused features [B, T-1, D]."""
        if self.config.ablation_confounder:
            return None
        return self.confounder(fused_global)

    def forward(self, batch: FrameBatch) -> ModelOutput:
        batch.validate()

        context_tokens = self.context_encoder.forward_tokens(batch.images)
        with torch.no_grad():
            target_tokens = self.target_encoder.forward_tokens(batch.images)

        encoded = self.encode_transitions(batch.images, context_tokens)
        dynamic, fused = encoded["dynamic"], encoded["fused"]
        z_task, z_exogenous = encoded["z_task"], encoded["z_exogenous"]
        P_task, P_exogenous = encoded["P_task"], encoded["P_exogenous"]

        # Step 5: Confounder GRU on patch-pooled fused features
        U = self.summarize_context(fused.mean(dim=2))

        # Step 6: Flatten spatiotemporal tokens for masking
        B, T_1, N_p, D = z_task.shape
        z_task_flat   = z_task.reshape(B, T_1 * N_p, D)
        target_flat   = target_tokens.reshape(B, T_1 * N_p, D)

        # Step 7: Spatiotemporal block masking
        z_task_masked, context_indices, mask_indices = self.mask_sampler(z_task_flat)

        # Step 8: Target patches at masked positions
        z_target = self._gather_tokens(target_flat, mask_indices)

        # Step 9: Causal predictor
        z_pred = self.predictor(z_task_masked, U, context_indices, mask_indices)

        if z_exogenous is not None:
            z_exogenous = z_exogenous.reshape(B, T_1 * N_p, D)

        return ModelOutput(
            z_pred=z_pred,
            target_patches=z_target,
            static_features=context_tokens,
            fused_features=fused,
            z_task=z_task_flat,
            dynamic_features=dynamic,
            z_exogenous=z_exogenous,
            P_task=P_task,
            P_exogenous=P_exogenous,
            U=U,
            mask_indices=mask_indices,
            context_indices=context_indices,
            losses=None,
        )