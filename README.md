# Membership Auditing beyond a Single Reference Population

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended. Import the package with `src` on `PYTHONPATH`.

## Generation

Synthetic images are drawn from a 32×32 convolutional generator. The checkpoint supplies the generator weights, the latent size, and the channel width. Returned images are unsigned 8-bit arrays in height-width-channel order.

```bash
python scripts/sample_images.py --checkpoint CHECKPOINT --n 1000 --out images.npz
```

Sampling requires PyTorch.

## Scores

Scores are computed from caller-supplied embeddings, text, or class probabilities. A larger value is stronger membership evidence, except training negative log-likelihood, which is kept as a loss. For a language-model loss or an NLL gap, pass `--negate` before comparing the score with an audit threshold.

- Max cosine is the highest cosine similarity between a query embedding and embeddings of synthetic images.
- Sequence NLL is the mean token cross-entropy under a causal language model.
- The NLL gap subtracts a fixed reference model's NLL from the audited model's NLL.
- Binary NLL is the negative log-likelihood of the true binary label. The matching log-probability score negates it.
- The likelihood ratio is the audited classifier's log probability of the true class minus the log of the mean probability under the reference classifiers.

```bash
python scripts/membership_scores.py cosine --queries QUERIES.npy --corpus CORPUS.npy --out scores.npy
python scripts/membership_scores.py nll --model MODEL --texts TEXTS.txt --out scores.npy
python scripts/membership_scores.py nll-gap --model MODEL --reference REFERENCE --texts TEXTS.txt --out scores.npy
python scripts/membership_scores.py binary-nll --probabilities PROB.npy --labels LABELS.npy --out scores.npy
python scripts/membership_scores.py log-prob --probabilities PROB.npy --labels LABELS.npy --out scores.npy
python scripts/membership_scores.py ratio --victim VICTIM.npy --reference REFS.npy --out scores.npy
```

Language-model scores require PyTorch and Transformers. A reference matrix has one row per reference classifier.

## License

MIT License. See [LICENSE](LICENSE).
