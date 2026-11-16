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

Scores are computed from caller-supplied embeddings, text, or class probabilities. A larger value is stronger membership evidence. The sign is fixed in the score definition. For a language-model loss or an NLL gap, pass `--negate` before comparing the score with an audit threshold. Logistic regression uses the negated training negative log-likelihood, which equals the log-probability of the true label.

- Max cosine is the highest cosine similarity between a query embedding and embeddings of synthetic images.
- Sequence NLL is the mean token cross-entropy under a causal language model.
- The NLL gap subtracts a fixed reference model's NLL from the audited model's NLL.
- Binary NLL is the negative log-likelihood of the true binary label. The audit score for logistic regression is its negation, saved by `reproduce.py logistic` as `member` and `nonmember`. `log-prob` computes that same quantity.
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

## Protocol

Training seeds are `0`, `1`, and `2`. For images and text, the private subset is drawn with NumPy seed `100 + seed` and the complement is the holdout. Logistic regression instead permutes the Adult table with `seed` itself. The wide residual network mixes a fixed label into its training and evaluation seeds; `python scripts/reproduce.py splits` prints those values.

| Setting | Private size | Non-members | Training |
|---|---:|---|---|
| Generator | 500 (also 1000, 5000) | all 10,000 CIFAR-10 test images | 20,000 hinge steps, batch 128, then 20,000 synthetic images |
| Language model | 512 (also 128, 256) | 500 test texts | 640 steps, length 256, from the public `gpt2` checkpoint |
| Logistic regression | 2,000 | the next 15,000 rows | 500 iterations |
| Classifier | 25,000 | 5,000 test images, and 5,000 members | 100 epochs; four reference models on the holdout |

```bash
python scripts/reproduce.py splits --setting generator --seed 0
python scripts/reproduce.py generator --data-root DATA --seed 0 --out generator.pt --scores scores.npz --download --score
python scripts/reproduce.py language --data-root DATA --seed 0 --out model --scores scores.npz
python scripts/reproduce.py logistic --seed 0 --scores scores.npz
python scripts/reproduce.py classifier --data-root DATA --seed 0 --scores scores.npz --download
```

Generator sampling needs PyTorch. Image settings also need torchvision, the language-model setting needs Transformers, and logistic regression needs scikit-learn. `--steps` and `--epochs` shorten a run without changing the split.

## License

MIT License. See [LICENSE](LICENSE).
