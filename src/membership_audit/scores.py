"""Membership scores used by the audit.

A larger score is stronger membership evidence, except training negative
log-likelihood, which is left as a loss. Loss-type language-model scores are
negated with ``negate`` before they are compared with a threshold.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def max_cosine(queries: np.ndarray, corpus: np.ndarray, batch: int = 512) -> np.ndarray:
    """Highest cosine similarity of each query embedding to a synthetic corpus."""
    queries = np.asarray(queries, dtype=np.float64)
    corpus = np.asarray(corpus, dtype=np.float64)
    if queries.ndim != 2 or corpus.ndim != 2:
        raise ValueError("queries and corpus must be matrices")
    if corpus.shape[0] == 0:
        raise ValueError("corpus is empty")

    def _normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norm, 1e-12)

    q = _normalize(queries)
    c = _normalize(corpus)
    out = np.empty(q.shape[0], dtype=np.float64)
    step = max(int(batch), 1)
    for i in range(0, q.shape[0], step):
        out[i : i + step] = (q[i : i + step] @ c.T).max(axis=1)
    return out


def negate(scores: np.ndarray) -> np.ndarray:
    """Flip a loss so that a larger value is stronger membership evidence."""
    return -np.asarray(scores, dtype=np.float64)


def sequence_nll(model, tokenizer, texts: Sequence[str], max_length: int, batch_size: int = 16) -> np.ndarray:
    """Mean token cross-entropy of each text under a causal language model."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for language-model scores") from exc

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    device = next(model.parameters()).device
    scores: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = [str(x) for x in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
            labels = input_ids[:, 1:]
            mask = attention_mask[:, 1:].bool()
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            denom = mask.sum(dim=1).clamp(min=1)
            sample_nll = (token_loss * mask).sum(dim=1) / denom
            scores.append(sample_nll.detach().cpu().numpy().astype(np.float64))
    if not scores:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(scores, axis=0)


def nll_gap(audited_nll: np.ndarray, reference_nll: np.ndarray) -> np.ndarray:
    """Audited-model NLL minus a fixed reference-model NLL."""
    audited = np.asarray(audited_nll, dtype=np.float64)
    reference = np.asarray(reference_nll, dtype=np.float64)
    if audited.shape != reference.shape:
        raise ValueError("audited and reference NLL must have the same shape")
    return audited - reference


def binary_nll(probability: np.ndarray, labels: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Negative log-likelihood of a binary label. ``probability`` is P(label = 1)."""
    p = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(labels, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError("probability and labels must have the same shape")
    return -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))


def true_label_logprob(probabilities: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Log probability of the true class.

    A vector is treated as P(label = 1) with binary labels. A matrix is a
    distribution over classes, and ``labels`` indexes the true column.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    if probs.ndim == 1:
        return -binary_nll(probs, y, eps=eps)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a vector or a matrix")
    if y.shape[0] != probs.shape[0]:
        raise ValueError("labels must match the number of rows")
    chosen = probs[np.arange(probs.shape[0]), y.astype(int)]
    return np.log(np.clip(chosen, eps, 1.0))


def likelihood_ratio(victim: np.ndarray, reference: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Log probability under the audited model minus the log mean reference probability.

    ``reference`` is either one probability per example, or a matrix whose
    rows are reference models and whose columns are examples.
    """
    victim = np.asarray(victim, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim == 2:
        if reference.shape[0] == 0 or reference.shape[1] != victim.shape[0]:
            raise ValueError("reference matrix must have shape (n_models, n_examples)")
        reference = reference.mean(axis=0)
    elif reference.shape != victim.shape:
        raise ValueError("reference probabilities must match the audited probabilities")
    return np.log(np.clip(victim, eps, 1.0)) - np.log(np.clip(reference, eps, 1.0))
