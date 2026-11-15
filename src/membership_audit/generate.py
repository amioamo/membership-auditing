"""Sample images from a 32×32 convolutional generator."""
from __future__ import annotations

from typing import Any

import numpy as np


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required to sample images") from exc
    return torch, nn


def build_generator(latent_dim: int = 128, base_channels: int = 64):
    """Convolutional generator from a latent vector to a 32×32 RGB image in [-1, 1]."""
    torch, nn = _torch()

    class Generator32(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(base_channels * 2, 3, 4, 2, 1, bias=False),
                nn.Tanh(),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z)

    return Generator32()


def load_generator(checkpoint: dict[str, Any] | str, device: str = "cpu"):
    """Load a generator from an in-memory checkpoint or a file path."""
    torch, _nn = _torch()
    payload = checkpoint
    if isinstance(checkpoint, str):
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "generator" not in payload:
        raise ValueError("Checkpoint must contain generator weights")
    image_size = int(payload.get("image_size", 32))
    if image_size != 32:
        raise ValueError(f"Only 32×32 images are supported, got {image_size}")
    latent_dim = int(payload.get("latent_dim", 128))
    base_channels = int(payload.get("g_base_channels", 64))
    generator = build_generator(latent_dim=latent_dim, base_channels=base_channels)
    generator.load_state_dict(payload["generator"])
    generator.to(device)
    generator.eval()
    return generator, latent_dim


def sample_images(
    generator,
    n: int,
    latent_dim: int,
    seed: int = 0,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Draw ``n`` uint8 images with shape ``(n, 32, 32, 3)``."""
    torch, _nn = _torch()
    if n <= 0:
        raise ValueError("n must be positive")
    torch.manual_seed(int(seed))
    generator.to(device)
    generator.eval()
    batches: list[np.ndarray] = []
    remaining = int(n)
    with torch.inference_mode():
        while remaining > 0:
            current = min(int(batch_size), remaining)
            z = torch.randn(current, latent_dim, 1, 1, device=device)
            images = generator(z)
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
            batches.append(images.permute(0, 2, 3, 1).detach().cpu().numpy())
            remaining -= current
    return np.concatenate(batches, axis=0)
