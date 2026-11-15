"""Draw images from a saved 32×32 generator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from membership_audit.generate import load_generator, sample_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample uint8 images from a generator checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    generator, latent_dim = load_generator(args.checkpoint, device=args.device)
    images = sample_images(
        generator,
        n=args.n,
        latent_dim=latent_dim,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
    )
    np.savez_compressed(args.out, images=images)


if __name__ == "__main__":
    main()
