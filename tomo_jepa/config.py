"""Configuration objects for ToMo-JEPA training."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    path = Path(value or default).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return str(path.resolve())


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def _run_output_path(name: str, suffix: str) -> str:
    run_dir = os.environ.get("RUN_DIR", "outputs").strip() or "outputs"
    return _env_path(name, str(Path(run_dir) / suffix))


@dataclass(frozen=True)
class ToMoJEPAConfig:
    # Dataset and dataloader parameters
    dataset_root: str = field(
        default_factory=lambda: _env_path("DATASET_ROOT", "data")
    )
    dataset_split: str = field(default_factory=lambda: os.environ.get("DATASET_SPLIT", "train"))
    history_length: int = 16
    stride: int = 1
    image_size: Tuple[int, int] = (256, 256)
    max_cached_episodes: int = 16
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 12))
    shuffle: bool = True
    num_workers: int = field(default_factory=lambda: _env_int("NUM_WORKERS", 4))

    # Trainable model components
    freeze_context_encoder: bool = True
    freeze_target_encoder: bool = True
    freeze_flow_estimator: bool = True
    freeze_motion_encoder: bool = False
    freeze_fusion: bool = False
    freeze_factorizer: bool = False
    freeze_confounder: bool = False
    freeze_predictor: bool = False

    # Ablation study flags
    ablation_motion_branch: bool = field(
        default_factory=lambda: _env_bool("ABLATION_MOTION_BRANCH", False)
    )  # True: disable the motion branch
    ablation_confounder: bool = field(
        default_factory=lambda: _env_bool("ABLATION_CONFOUNDER", False)
    )  # True: disable the confounder branch
    ablation_factorizer: bool = field(
        default_factory=lambda: _env_bool("ABLATION_FACTORIZER", False)
    )  # True: disable the factorizer branch

    # Drive-JEPA ViT-L checkpoint and V-JEPA2 root path
    vitl_checkpoint_path: str = field(
        default_factory=lambda: _env_path(
            "VITL_CHECKPOINT_PATH", "checkpoints/vitl_merge_3dataset_e50.pt"
        )
    )
    vjepa2_root: str = field(default_factory=lambda: _env_path("VJEPA2_ROOT", "third_party/vjepa2"))

    # FlowFormer++ checkpoint and root path
    motion_estimator_checkpoint_path: str = field(
        default_factory=lambda: _env_path(
            "FLOWFORMER_CHECKPOINT_PATH",
            "checkpoints/kitti.pth",
        )
    )
    flowformer_root: str = field(
        default_factory=lambda: _env_path("FLOWFORMER_ROOT", "third_party/flowformer")
    )

    # Motion encoder parameters
    flow_channels: int = 2

    # Fusion parameters
    fusion_num_heads: int = 16
    fusion_scale: float = 1.0

    # Latent Factorizer parameters
    task_dim: int = 512
    enforce_orthogonality: bool = True

    # Confounder parameters
    confounder_dim: int = 128

    # Predictor parameters
    predictor_embed_dim: int = 384
    predictor_depth: int = 12
    predictor_num_heads: int = 12
    predictor_num_mask_tokens: int = 10
    predictor_confounder_mode: str = "add"

    # Masking parameters
    mask_ratio: float = 0.7

    # ToMo-JEPA training parameters
    output_checkpoint_path: str = field(
        default_factory=lambda: _run_output_path(
            "OUTPUT_CHECKPOINT_PATH", "checkpoints/tomo.pt"
        )
    )
    output_log_dir: str = field(
        default_factory=lambda: _run_output_path("OUTPUT_LOG_DIR", "logs")
    )
    pretrained: bool = True
    resume_checkpoint_path: str = field(
        default_factory=lambda: (
            _env_path("RESUME_CHECKPOINT", "")
            if os.environ.get("RESUME_CHECKPOINT", "").strip()
            else ""
        )
    )
    latent_dim: int = 1024
    num_epochs: int = field(default_factory=lambda: _env_int("EPOCHS", 10))
    n_steps_per_epoch: int = 1000
    learning_rate: float = 0.0002
    target_ema_momentum: float = 0.99925
    # Loss weights
    jepa_loss_weight: float = 1.0
    orthogonality_loss_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.history_length < 2:
            raise ValueError("history_length must be at least 2 for consecutive frame clips")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if len(self.image_size) != 2 or any(size < 1 for size in self.image_size):
            raise ValueError("image_size must be a positive (width, height) pair")
        if self.latent_dim != 1024:
            raise ValueError("latent_dim must be 1024 to match the Drive-JEPA ViT-L encoder")
        if self.fusion_num_heads < 1:
            raise ValueError("fusion_num_heads must be positive")
        if not 0.0 < self.target_ema_momentum <= 1.0:
            raise ValueError("target_ema_momentum must be in (0, 1]")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if self.predictor_embed_dim < 1 or self.predictor_depth < 1 or self.predictor_num_heads < 1:
            raise ValueError("predictor dimensions must be positive")
        if self.predictor_num_mask_tokens < 1:
            raise ValueError("predictor_num_mask_tokens must be positive")
        if not self.dataset_root:
            raise ValueError("dataset_root must not be empty")
        if not self.dataset_split:
            raise ValueError("dataset_split must not be empty")
        if not self.vitl_checkpoint_path:
            raise ValueError("vitl_checkpoint_path must not be empty")
        if not self.vjepa2_root:
            raise ValueError("vjepa2_root must not be empty")
        if not self.output_checkpoint_path:
            raise ValueError("output_checkpoint_path must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be positive")
