"""PyTorch data loading for the common ``tomo`` episode format."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ..config import ToMoJEPAConfig
from ..contracts import FrameBatch


@dataclass(frozen=True)
class EpisodeWindow:
    dataset_root: Path
    episode_path: Path
    start_index: int


class ToMoEpisodeDataset(Dataset[dict[str, torch.Tensor]]):
    """Load fixed-length image windows from converted ToMo episodes.

    ``dataset_root`` can identify a single converted source:

    .. code-block:: text

        data/
          episodes/<split>/<log_name>.npz
          images/<split>/<log_name>/<token>.jpg

    Or it can identify an aggregate directory, whose direct child
    directories with an ``episodes/`` directory are discovered as sources.
    Each episode must contain ``timestamps_us``, ``image_paths`` and
    ``can_bus`` arrays with the same leading frame dimension. Image paths
    are relative to the source dataset root. Unreadable windows are skipped per
    worker; access fails explicitly if no readable windows remain.

    With ``future_length > 0`` each window spans ``history_length + future_length``
    frames: images and timestamps cover the history only, while
    ``can_bus`` also covers the future frames (used as ground truth).
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        history_length: int = 4,
        stride: int = 1,
        image_size: int | tuple[int, int] = (256, 256),
        max_cached_episodes: int = 8,
        future_length: int = 0,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if future_length < 0:
            raise ValueError("future_length must be non-negative")
        if stride < 1:
            raise ValueError("stride must be positive")
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if len(image_size) != 2 or any(size < 1 for size in image_size):
            raise ValueError("image_size must be a positive (width, height) pair")
        if max_cached_episodes < 1:
            raise ValueError("max_cached_episodes must be positive")

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.history_length = history_length
        self.future_length = future_length
        self.stride = stride
        self.image_size = image_size
        self.max_cached_episodes = max_cached_episodes
        self._episode_cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._windows = self._discover_windows()
        self._invalid_windows: set[int] = set()
        if not self._windows:
            raise ValueError(
                f"no episodes in {self.dataset_root} for split={split!r} "
                f"with history_length={history_length}"
            )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self._windows)
        if not 0 <= index < len(self._windows):
            raise IndexError(f"ToMo dataset index out of range: {index}")

        last_error: OSError | ValueError | None = None
        for offset in range(len(self._windows)):
            window_index = (index + offset) % len(self._windows)
            if window_index in self._invalid_windows:
                continue

            window = self._windows[window_index]
            try:
                episode = self._load_episode(window.episode_path)
                stop_index = window.start_index + self.history_length
                can_bus_stop = stop_index + self.future_length
                image_paths = episode["image_paths"][window.start_index:stop_index]
                images = torch.stack(
                    [self._load_image(window.dataset_root / str(path)) for path in image_paths]
                )
                can_bus = torch.from_numpy(
                    episode["can_bus"][window.start_index:can_bus_stop].astype(np.float32, copy=False)
                )
                timestamps_us = torch.from_numpy(
                    episode["timestamps_us"][window.start_index:stop_index].astype(np.int64, copy=False)
                )
            except (OSError, ValueError) as error:
                self._invalid_windows.add(window_index)
                last_error = error
                print(
                    f"Warning: Skipping unreadable ToMo window {window_index} "
                    f"({window.episode_path}, start={window.start_index}): {error}"
                )
                continue

            return {
                "images": images,
                "can_bus": can_bus,
                "timestamps_us": timestamps_us,
            }

        if last_error is not None:
            raise RuntimeError(
                f"No readable ToMo windows remain in {self.dataset_root} for split={self.split!r}"
            ) from last_error
        raise RuntimeError(f"No readable ToMo windows remain in {self.dataset_root}")

    def _discover_windows(self) -> list[EpisodeWindow]:
        windows: list[EpisodeWindow] = []
        for source_root in self._source_roots():
            for episode_path in sorted((source_root / "episodes" / self.split).glob("*.npz")):
                episode = self._load_episode(episode_path)
                frame_count = len(episode["timestamps_us"])
                for start_index in range(0, frame_count - self.history_length - self.future_length + 1, self.stride):
                    windows.append(EpisodeWindow(source_root, episode_path, start_index))
        return windows

    def _source_roots(self) -> list[Path]:
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"ToMo dataset root does not exist: {self.dataset_root}")
        if (self.dataset_root / "episodes").is_dir():
            return [self.dataset_root]
        source_roots = sorted(
            path for path in self.dataset_root.iterdir() if path.is_dir() and (path / "episodes").is_dir()
        )
        if not source_roots:
            raise FileNotFoundError(
                f"expected episodes/ in {self.dataset_root} or one of its immediate child directories"
            )
        return source_roots

    def _load_episode(self, episode_path: Path) -> dict[str, np.ndarray]:
        cached = self._episode_cache.get(episode_path)
        if cached is not None:
            self._episode_cache.move_to_end(episode_path)
            return cached

        with np.load(episode_path, allow_pickle=False) as raw_episode:
            required = {"timestamps_us", "image_paths", "can_bus"}
            missing = required.difference(raw_episode.files)
            if missing:
                raise ValueError(f"invalid ToMo episode {episode_path}: missing {sorted(missing)}")
            episode = {name: raw_episode[name] for name in required}

        frame_count = len(episode["timestamps_us"])
        if any(len(episode[name]) != frame_count for name in ("image_paths", "can_bus")):
            raise ValueError(f"invalid ToMo episode {episode_path}: inconsistent frame dimensions")
        if episode["can_bus"].ndim != 2:
            raise ValueError(f"invalid ToMo episode {episode_path}: can_bus must be a matrix")

        self._episode_cache[episode_path] = episode
        if len(self._episode_cache) > self.max_cached_episodes:
            self._episode_cache.popitem(last=False)
        return episode

    def _load_image(self, image_path: Path) -> torch.Tensor:
        if not image_path.is_file():
            raise FileNotFoundError(f"ToMo image does not exist: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = ImageOps.fit(
                image,
                self.image_size,
                method=Image.Resampling.BILINEAR,
            )
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def collate_tomo_samples(samples: Sequence[dict[str, torch.Tensor]]) -> FrameBatch:
    """Collate ToMo dataset samples into the pipeline's ``FrameBatch`` contract."""
    if not samples:
        raise ValueError("cannot collate an empty list of ToMo samples")
    batch = FrameBatch(
        images=torch.stack([sample["images"] for sample in samples]),
    )
    batch.validate()
    return batch


def future_trajectory_ego_frame(can_bus: torch.Tensor, history_length: int) -> torch.Tensor:
    """can_bus [B, history+H, 18] -> future positions [B, H, 2] (meters) in the ego frame at the last observed frame."""
    now = history_length - 1
    delta = can_bus[:, history_length:, 0:2] - can_bus[:, now:now + 1, 0:2]
    w, x, y, z = can_bus[:, now, 3], can_bus[:, now, 4], can_bus[:, now, 5], can_bus[:, now, 6]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).unsqueeze(-1)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    forward = cos * delta[..., 0] + sin * delta[..., 1]
    left = -sin * delta[..., 0] + cos * delta[..., 1]
    return torch.stack([forward, left], dim=-1)


def make_tomo_dataloader(
    config: ToMoJEPAConfig,
    distributed: bool = False,
) -> DataLoader[FrameBatch]:
    """Create the configured DataLoader yielding ``FrameBatch(images)``."""
    dataset = ToMoEpisodeDataset(
        dataset_root=config.dataset_root,
        split=config.dataset_split,
        history_length=config.history_length,
        stride=config.stride,
        image_size=config.image_size,
        max_cached_episodes=config.max_cached_episodes,
    )
    sampler = DistributedSampler(dataset, shuffle=config.shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle and sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_tomo_samples,
    )