"""Train one audited model and write its membership scores.

Data are read from a directory supplied by the caller, or downloaded there
on request. Defaults follow the splits and seeds in ``protocol.SETTINGS``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from membership_audit.fit import (  # noqa: E402
    class_probability,
    classifier_scores,
    finetune_language,
    fit_logistic,
    logistic_scores,
    train_classifier,
    train_generator,
)
from membership_audit.generate import load_generator, sample_images  # noqa: E402
from membership_audit.protocol import (  # noqa: E402
    SEEDS,
    SETTINGS,
    classifier_seeds,
    language_nonmember_indices,
    private_split,
    table_split,
)
from membership_audit.scores import max_cosine, nll_gap, sequence_nll  # noqa: E402


def _splits(setting: str, seed: int, n_private: int | None) -> None:
    cfg = SETTINGS[setting]
    population = int(cfg.get("population_size", 0))
    n_private = int(n_private if n_private is not None else cfg["n_private"])
    print(f"setting={setting} seed={seed} seeds={list(SEEDS)} n_private={n_private}")
    if setting == "logistic":
        print("split=permutation of the Adult table")
        print(f"holdout={cfg['holdout_size']}")
        return
    if population:
        private, holdout = private_split(population, n_private, seed)
        print(f"split_seed={100 + seed}")
        print(f"private={private.shape[0]} holdout={holdout.shape[0]}")
        print(f"private_head={private[:8].tolist()}")
    if setting == "language":
        print("nonmembers=500 test texts")
        print(f"nonmember_seed_label=text_pilot_npriv{n_private}_seed{seed}")
    if setting == "classifier":
        drawn = classifier_seeds(seed, n_private)
        print(f"seed_label={drawn['label']}")
        print(f"victim_seed={drawn['victim']}")
        print(f"member_eval={cfg['member_eval_size']} nonmember_eval={cfg['nonmember_size']}")
        print(f"reference_models={cfg['reference_models']}")


def _cifar(root: str, train: bool, download: bool):
    from torchvision import datasets

    return datasets.CIFAR10(root=root, train=train, download=download)


def _image_batch(
    dataset,
    indices: np.ndarray,
    mean,
    std,
    augment: bool,
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = False,
):
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    steps = []
    if augment:
        steps.extend(
            [
                transforms.RandomCrop(32, padding=int(SETTINGS["classifier"]["crop_padding"])),
                transforms.RandomHorizontalFlip(),
            ]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    dataset.transform = transforms.Compose(steps)
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def _generator(args: argparse.Namespace) -> None:
    cfg = SETTINGS["generator"]
    n_private = int(args.n_private or cfg["n_private"])
    train_set = _cifar(args.data_root, train=True, download=args.download)
    private, _holdout = private_split(len(train_set), n_private, args.seed)
    loader = _image_batch(
        train_set,
        private,
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        augment=False,
        batch_size=int(cfg["batch_size"]),
        drop_last=True,
        shuffle=True,
    )
    checkpoint = train_generator(loader, steps=args.steps, seed=args.seed, device=args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(checkpoint, out)
    generator, latent = load_generator(checkpoint, device=args.device or "cpu")
    synthetic = sample_images(
        generator,
        n=int(args.samples or cfg["sample_count"]),
        latent_dim=latent,
        seed=int(cfg["sample_seed"]),
        device=args.device or "cpu",
    )
    np.savez_compressed(out.with_suffix(".images.npz"), images=synthetic, private_indices=private)


def _embed(images_uint8: np.ndarray) -> np.ndarray:
    import torch
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = weights.transforms()
    from PIL import Image

    rows = []
    with torch.inference_mode():
        for image in images_uint8:
            tensor = transform(Image.fromarray(image)).unsqueeze(0)
            rows.append(model(tensor).detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _uint8_from_dataset(dataset, indices: np.ndarray) -> np.ndarray:
    images = []
    for index in indices:
        image, _label = dataset[int(index)]
        images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images, axis=0)


def _generator_scores(args: argparse.Namespace) -> None:
    _generator(args)
    images = np.load(Path(args.out).with_suffix(".images.npz"))
    train_set = _cifar(args.data_root, train=True, download=False)
    train_set.transform = None
    test_set = _cifar(args.data_root, train=False, download=args.download)
    test_set.transform = None
    private = images["private_indices"]
    synthetic = _embed(images["images"])
    members = _embed(_uint8_from_dataset(train_set, private))
    nonmembers = _embed(_uint8_from_dataset(test_set, np.arange(len(test_set))))
    np.savez_compressed(
        args.scores,
        member_scores=max_cosine(members, synthetic),
        nonmember_scores=max_cosine(nonmembers, synthetic),
        private_indices=private,
    )


def _language(args: argparse.Namespace) -> None:
    from sklearn.datasets import fetch_20newsgroups

    cfg = SETTINGS["language"]
    n_private = int(args.n_private or cfg["n_private"])
    remove = ("headers", "footers", "quotes")
    train = fetch_20newsgroups(subset="train", remove=remove, shuffle=False, data_home=args.data_root)
    test = fetch_20newsgroups(subset="test", remove=remove, shuffle=False, data_home=args.data_root)
    private, _holdout = private_split(len(train.data), n_private, args.seed)
    texts = [train.data[int(i)] for i in private]
    model, tokenizer = finetune_language(texts, seed=args.seed, steps=args.steps, out_dir=args.out)
    nonmember_index = language_nonmember_indices(len(test.data), n_private, args.seed, int(cfg["nonmember_size"]))
    member_texts = texts
    nonmember_texts = [test.data[int(i)] for i in nonmember_index]
    reference_name = cfg["model_name"]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reference_tokenizer = AutoTokenizer.from_pretrained(reference_name)
    if reference_tokenizer.pad_token is None:
        reference_tokenizer.pad_token = reference_tokenizer.eos_token
    reference = AutoModelForCausalLM.from_pretrained(reference_name)
    reference.config.pad_token_id = reference_tokenizer.pad_token_id
    reference.to(next(model.parameters()).device)
    max_length = int(cfg["max_length"])
    batch = int(cfg["batch_size"])

    def gap(group: list[str]) -> tuple[np.ndarray, np.ndarray]:
        audited = sequence_nll(model, tokenizer, group, max_length, batch)
        base = sequence_nll(reference, reference_tokenizer, group, max_length, batch)
        return -audited, -nll_gap(audited, base)

    member_nll, member_gap = gap(member_texts)
    nonmember_nll, nonmember_gap = gap(nonmember_texts)
    np.savez_compressed(
        args.scores,
        nll_member=member_nll,
        nll_nonmember=nonmember_nll,
        ratio_member=member_gap,
        ratio_nonmember=nonmember_gap,
        private_indices=private,
        nonmember_indices=nonmember_index,
    )


def _logistic(args: argparse.Namespace) -> None:
    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import fetch_openml
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cfg = SETTINGS["logistic"]
    frame = fetch_openml("adult", version=2, as_frame=True, parser="auto").frame
    labels = (frame["class"] == ">50K").astype(int).to_numpy()
    features = frame.drop(columns=["class"])
    private, holdout = table_split(len(labels), int(cfg["n_private"]), int(cfg["holdout_size"]), args.seed)
    numeric = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    model = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), numeric),
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
                    ]
                ),
            ),
        ]
    )
    transformed = model.fit_transform(features)
    fitted, member_probability = fit_logistic(transformed[private], labels[private], seed=args.seed)
    holdout_probability = fitted.predict_proba(transformed[holdout])[:, 1]
    scores = logistic_scores(member_probability, labels[private], holdout_probability, labels[holdout])
    occupation = features.iloc[holdout]["occupation"].astype("string").fillna("Unknown").to_numpy()
    values, counts = np.unique(occupation, return_counts=True)
    top = values[np.argsort(-counts)[:5]]
    groups = np.array([{name: i for i, name in enumerate(top)}.get(str(name), 5) for name in occupation])
    np.savez_compressed(args.scores, private_indices=private, holdout_indices=holdout, occupation=groups, **scores)


def _loader(dataset, indices: np.ndarray, train: bool):
    cfg = SETTINGS["classifier"]
    return _image_batch(
        dataset,
        indices,
        cfg["mean"],
        cfg["std"],
        augment=train,
        batch_size=int(cfg["batch_size"]),
        drop_last=False,
        shuffle=train,
    )


def _classifier(args: argparse.Namespace) -> None:
    cfg = SETTINGS["classifier"]
    n_private = int(args.n_private or cfg["n_private"])
    train_set = _cifar(args.data_root, train=True, download=args.download)
    test_set = _cifar(args.data_root, train=False, download=args.download)
    private, holdout = private_split(len(train_set), n_private, args.seed)
    drawn = classifier_seeds(args.seed, n_private)
    from membership_audit.protocol import choose

    victim = train_classifier(
        _loader(train_set, private, train=True),
        seed=int(drawn["victim"]),
        epochs=args.epochs,
        device=args.device,
    )
    member_index = choose(private, int(cfg["member_eval_size"]), int(drawn["member_eval"]))
    nonmember_index = choose(np.arange(len(test_set)), int(cfg["nonmember_size"]), int(drawn["nonmember_eval"]))
    victim_member = class_probability(victim, _loader(train_set, member_index, train=False))
    victim_nonmember = class_probability(victim, _loader(test_set, nonmember_index, train=False))
    reference_member = []
    reference_nonmember = []
    for index in range(int(cfg["reference_models"])):
        subset = choose(holdout, min(n_private, holdout.shape[0]), int(drawn["reference_subset"][index]))
        reference = train_classifier(
            _loader(train_set, subset, train=True),
            seed=int(drawn["reference_train"][index]),
            epochs=args.epochs,
            device=args.device,
        )
        reference_member.append(class_probability(reference, _loader(train_set, member_index, train=False)))
        reference_nonmember.append(class_probability(reference, _loader(test_set, nonmember_index, train=False)))
    member_rows = np.stack(reference_member, axis=0)
    nonmember_rows = np.stack(reference_nonmember, axis=0)
    member_scores = classifier_scores(victim_member, member_rows)
    nonmember_scores = classifier_scores(victim_nonmember, nonmember_rows)
    np.savez_compressed(
        args.scores,
        logprob_member=member_scores["logprob"],
        logprob_nonmember=nonmember_scores["logprob"],
        ratio_member=member_scores["ratio"],
        ratio_nonmember=nonmember_scores["ratio"],
        private_indices=private,
        member_indices=member_index,
        nonmember_indices=nonmember_index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an audited model and write scores.")
    sub = parser.add_subparsers(dest="command", required=True)

    splits = sub.add_parser("splits")
    splits.add_argument("--setting", required=True, choices=tuple(SETTINGS))
    splits.add_argument("--seed", type=int, default=0)
    splits.add_argument("--n-private", type=int)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--n-private", type=int)
    common.add_argument("--scores", required=True)
    common.add_argument("--device", default=None)

    generator = sub.add_parser("generator", parents=[common])
    generator.add_argument("--data-root", required=True)
    generator.add_argument("--out", required=True)
    generator.add_argument("--steps", type=int)
    generator.add_argument("--samples", type=int)
    generator.add_argument("--download", action="store_true")
    generator.add_argument("--score", action="store_true")

    language = sub.add_parser("language", parents=[common])
    language.add_argument("--data-root", required=True)
    language.add_argument("--out", required=True)
    language.add_argument("--steps", type=int)

    logistic = sub.add_parser("logistic", parents=[common])

    classifier = sub.add_parser("classifier", parents=[common])
    classifier.add_argument("--data-root", required=True)
    classifier.add_argument("--epochs", type=int)
    classifier.add_argument("--download", action="store_true")

    args = parser.parse_args()
    if args.command == "splits":
        _splits(args.setting, args.seed, args.n_private)
    elif args.command == "generator" and args.score:
        _generator_scores(args)
    elif args.command == "generator":
        _generator(args)
    elif args.command == "language":
        _language(args)
    elif args.command == "logistic":
        _logistic(args)
    elif args.command == "classifier":
        _classifier(args)


if __name__ == "__main__":
    main()
