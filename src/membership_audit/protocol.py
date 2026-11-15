"""Splits and training settings for the four audited models.

Private indices for images and text are a simple random sample. The generator
seed is ``split_offset + seed`` with ``split_offset`` equal to 100. The same
integer ``seed`` is then reused as the training seed, except for the wide
residual network, whose training and evaluation draws mix a fixed label into
the seed. Logistic regression permutes the table with ``seed`` directly.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

SPLIT_OFFSET = 100
SEEDS = (0, 1, 2)

SETTINGS: dict[str, dict[str, Any]] = {
    "generator": {
        "population": "CIFAR-10 training images",
        "population_size": 50_000,
        "n_private": 500,
        "n_private_extra": (1000, 5000),
        "nonmembers": "CIFAR-10 test images",
        "nonmember_size": 10_000,
        "train_steps": 20_000,
        "batch_size": 128,
        "latent_dim": 128,
        "generator_channels": 64,
        "discriminator_channels": 64,
        "learning_rate": 2e-4,
        "beta1": 0.5,
        "beta2": 0.999,
        "critic_steps": 1,
        "sample_count": 20_000,
        "sample_seed": 0,
        "horizontal_flip": False,
    },
    "language": {
        "population": "20 Newsgroups training texts, headers, footers, and quotes removed",
        "n_private": 512,
        "n_private_extra": (128, 256),
        "nonmembers": "20 Newsgroups test texts",
        "nonmember_size": 500,
        "model_name": "gpt2",
        "epochs": 5,
        "batch_size": 4,
        "max_length": 256,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "max_steps": 640,
    },
    "logistic": {
        "population": "UCI Adult, OpenML adult version 2",
        "n_private": 2000,
        "holdout_size": 15_000,
        "max_iter": 500,
    },
    "classifier": {
        "population": "CIFAR-10 training images",
        "population_size": 50_000,
        "n_private": 25_000,
        "nonmembers": "CIFAR-10 test images",
        "member_eval_size": 5000,
        "nonmember_size": 5000,
        "reference_models": 4,
        "reference_seed_offset": 10_000,
        "depth": 28,
        "widen_factor": 2,
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "crop_padding": 4,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
}


def mix_seed(*parts: object) -> int:
    """Integer seed mixed from the same fields used for subset draws."""
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def private_split(
    population_size: int,
    n_private: int,
    seed: int,
    offset: int = SPLIT_OFFSET,
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted private indices and the complementary holdout."""
    if not 0 < n_private <= population_size:
        raise ValueError("n_private must lie strictly between 0 and the population size")
    rng = np.random.default_rng(offset + int(seed))
    private = np.sort(rng.choice(population_size, size=n_private, replace=False))
    holdout = np.setdiff1d(np.arange(population_size, dtype=np.int64), private)
    return private.astype(np.int64), holdout.astype(np.int64)


def choose(pool: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Sorted without-replacement sample."""
    pool = np.asarray(pool, dtype=np.int64)
    if size > pool.shape[0]:
        raise ValueError("sample size exceeds the pool")
    if size == pool.shape[0]:
        return np.sort(pool)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(pool, size=size, replace=False)).astype(np.int64)


def table_split(n_rows: int, n_private: int, holdout_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Private prefix and following holdout of one permutation. Used for Adult."""
    if n_private + holdout_size > n_rows:
        raise ValueError("private and holdout exceed the table")
    order = np.random.default_rng(int(seed)).permutation(n_rows)
    private = order[:n_private]
    holdout = order[n_private : n_private + holdout_size]
    return private.astype(np.int64), holdout.astype(np.int64)


def language_nonmember_indices(test_size: int, n_private: int, seed: int, n_eval: int = 500) -> np.ndarray:
    """Test-set texts scored as non-members."""
    label = f"text_pilot_npriv{int(n_private)}_seed{int(seed)}"
    return choose(np.arange(test_size, dtype=np.int64), n_eval, mix_seed("text_eval", label, n_private, seed))


def classifier_seeds(seed: int, n_private: int = 25_000) -> dict[str, Any]:
    """Training and evaluation seeds for one wide-residual-network replicate."""
    label = f"rmia_probe_npriv{int(n_private)}_seed{int(seed)}"
    offset = int(SETTINGS["classifier"]["reference_seed_offset"])
    return {
        "label": label,
        "victim": mix_seed(label, "victim", seed),
        "member_eval": mix_seed(label, "member_eval", seed),
        "nonmember_eval": mix_seed(label, "nonmember_eval", seed),
        "reference_subset": [mix_seed(label, "ref_subset", offset, i) for i in range(4)],
        "reference_train": [mix_seed(label, "ref_model", offset, i) for i in range(4)],
    }
