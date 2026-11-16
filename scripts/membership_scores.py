"""Compute the membership scores consumed by the audit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from membership_audit.scores import (
    binary_nll,
    likelihood_ratio,
    max_cosine,
    nll_gap,
    sequence_nll,
    true_label_logprob,
)


def _load_array(path: str) -> np.ndarray:
    array = np.load(path)
    if isinstance(array, np.lib.npyio.NpzFile):
        if "scores" in array.files:
            return np.asarray(array["scores"])
        if len(array.files) == 1:
            return np.asarray(array[array.files[0]])
        raise KeyError(f"Array file has keys {array.files}")
    return np.asarray(array)


def _save(path: str, scores: np.ndarray) -> None:
    np.save(path, np.asarray(scores, dtype=np.float64))


def _read_texts(path: str) -> list[str]:
    return [line.rstrip("\n") for line in Path(path).read_text().splitlines() if line.strip()]


def _load_causal_lm(model_id: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("PyTorch and Transformers are required for language-model scores") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.config.pad_token_id = tokenizer.pad_token_id
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Membership scores.")
    sub = parser.add_subparsers(dest="command", required=True)

    cosine = sub.add_parser("cosine")
    cosine.add_argument("--queries", required=True)
    cosine.add_argument("--corpus", required=True)
    cosine.add_argument("--out", required=True)
    cosine.add_argument("--batch", type=int, default=512)

    nll = sub.add_parser("nll")
    nll.add_argument("--model", required=True)
    nll.add_argument("--texts", required=True)
    nll.add_argument("--out", required=True)
    nll.add_argument("--max-length", type=int, default=256)
    nll.add_argument("--batch-size", type=int, default=16)
    nll.add_argument("--negate", action="store_true")

    gap = sub.add_parser("nll-gap")
    gap.add_argument("--model", required=True)
    gap.add_argument("--reference", required=True)
    gap.add_argument("--texts", required=True)
    gap.add_argument("--out", required=True)
    gap.add_argument("--max-length", type=int, default=256)
    gap.add_argument("--batch-size", type=int, default=16)
    gap.add_argument("--negate", action="store_true")

    binary = sub.add_parser("binary-nll")
    binary.add_argument("--probabilities", required=True)
    binary.add_argument("--labels", required=True)
    binary.add_argument("--out", required=True)

    logprob = sub.add_parser("log-prob")
    logprob.add_argument("--probabilities", required=True)
    logprob.add_argument("--labels", required=True)
    logprob.add_argument("--out", required=True)

    ratio = sub.add_parser("ratio")
    ratio.add_argument("--victim", required=True)
    ratio.add_argument("--reference", required=True)
    ratio.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "cosine":
        scores = max_cosine(_load_array(args.queries), _load_array(args.corpus), batch=args.batch)
    elif args.command == "nll":
        model, tokenizer = _load_causal_lm(args.model)
        scores = sequence_nll(model, tokenizer, _read_texts(args.texts), args.max_length, args.batch_size)
        if args.negate:
            scores = -scores
    elif args.command == "nll-gap":
        audited, audited_tokenizer = _load_causal_lm(args.model)
        reference, reference_tokenizer = _load_causal_lm(args.reference)
        texts = _read_texts(args.texts)
        scores = nll_gap(
            sequence_nll(audited, audited_tokenizer, texts, args.max_length, args.batch_size),
            sequence_nll(reference, reference_tokenizer, texts, args.max_length, args.batch_size),
        )
        if args.negate:
            scores = -scores
    elif args.command == "binary-nll":
        scores = binary_nll(_load_array(args.probabilities), _load_array(args.labels))
    elif args.command == "log-prob":
        scores = true_label_logprob(_load_array(args.probabilities), _load_array(args.labels))
    elif args.command == "ratio":
        scores = likelihood_ratio(_load_array(args.victim), _load_array(args.reference))
    else:
        raise ValueError(args.command)
    _save(args.out, scores)


if __name__ == "__main__":
    main()
