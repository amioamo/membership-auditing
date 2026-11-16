"""Train the four audited models and return membership scores.

Callers supply the records. Image and text loaders stay outside this module.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .generate import build_discriminator, build_generator
from .protocol import SETTINGS
from .scores import binary_nll, likelihood_ratio


def set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def preferred_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _init_gan(module) -> None:
    import torch.nn as nn

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def train_generator(loader: Iterable, steps: int | None = None, seed: int = 0, device: str | None = None) -> dict:
    """Hinge training of the 32×32 generator.

    ``loader`` yields image batches already scaled to ``[-1, 1]``, shape
    ``(batch, 3, 32, 32)``. The returned checkpoint can be sampled with
    ``load_generator``.
    """
    import torch
    import torch.nn.functional as F

    cfg = SETTINGS["generator"]
    steps = int(cfg["train_steps"] if steps is None else steps)
    set_seed(seed)
    device = device or preferred_device()
    generator = build_generator(cfg["latent_dim"], cfg["generator_channels"]).to(device)
    discriminator = build_discriminator(cfg["discriminator_channels"]).to(device)
    generator.apply(_init_gan)
    discriminator.apply(_init_gan)
    opt_g = torch.optim.Adam(
        generator.parameters(), lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"])
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(), lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"])
    )
    latent = int(cfg["latent_dim"])
    critic_steps = int(cfg["critic_steps"])
    step = 0
    generator.train()
    discriminator.train()
    while step < steps:
        progressed = False
        for batch in loader:
            progressed = True
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)
            batch_size = images.shape[0]
            noise = torch.randn(batch_size, latent, 1, 1, device=device)
            with torch.no_grad():
                fake = generator(noise)
            loss_d = F.relu(1.0 - discriminator(images)).mean() + F.relu(1.0 + discriminator(fake)).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            if (step + 1) % critic_steps == 0:
                noise = torch.randn(batch_size, latent, 1, 1, device=device)
                loss_g = -discriminator(generator(noise)).mean()
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()
            step += 1
            if step >= steps:
                break
        if not progressed:
            raise ValueError("training loader is empty")
    generator.eval()
    return {
        "generator": generator.state_dict(),
        "latent_dim": latent,
        "g_base_channels": int(cfg["generator_channels"]),
        "image_size": 32,
    }


def finetune_language(texts: list[str], seed: int = 0, steps: int | None = None, out_dir: str | None = None):
    """Fine-tune the causal language model on the private texts."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = SETTINGS["language"]
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
    model.config.pad_token_id = tokenizer.pad_token_id
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=int(cfg["max_length"]),
        return_tensors="pt",
    )

    class _Texts(Dataset):
        def __len__(self) -> int:
            return int(encoded["input_ids"].shape[0])

        def __getitem__(self, index: int):
            return {
                "input_ids": encoded["input_ids"][index],
                "attention_mask": encoded["attention_mask"][index],
            }

    loader = DataLoader(_Texts(), batch_size=int(cfg["batch_size"]), shuffle=True)
    device = preferred_device()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    limit = int(cfg["epochs"]) * max(1, len(loader))
    if steps is not None:
        limit = min(limit, int(steps))
    elif int(cfg["max_steps"]) > 0:
        limit = min(limit, int(cfg["max_steps"]))
    model.train()
    seen = 0
    while seen < limit:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(labels=batch["input_ids"], **batch).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += 1
            if seen >= limit:
                break
    model.eval()
    if out_dir is not None:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    return model, tokenizer


def fit_logistic(features, labels, seed: int = 0):
    """Logistic regression. Returns P(label = 1) for the fitted rows."""
    from sklearn.linear_model import LogisticRegression

    cfg = SETTINGS["logistic"]
    model = LogisticRegression(max_iter=int(cfg["max_iter"]), random_state=int(seed))
    model.fit(features, labels)
    return model, model.predict_proba(features)[:, 1]


def logistic_scores(member_probability, member_labels, holdout_probability, holdout_labels) -> dict[str, np.ndarray]:
    """Low-loss logistic score: negated training NLL, equal to the true-label log-probability.

    The sign is fixed by this definition. Larger values are stronger membership evidence.
    """
    member = -binary_nll(member_probability, member_labels)
    holdout = -binary_nll(holdout_probability, holdout_labels)
    return {
        "member": member,
        "nonmember": holdout,
    }


def build_wide_resnet(num_classes: int = 10):
    """Wide residual network used for the CIFAR-10 classifier."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = SETTINGS["classifier"]
    depth = int(cfg["depth"])
    widen = int(cfg["widen_factor"])

    class BasicBlock(nn.Module):
        def __init__(self, in_planes: int, out_planes: int, stride: int, dropout_rate: float = 0.0) -> None:
            super().__init__()
            self.equal_in_out = in_planes == out_planes
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
            self.dropout_rate = float(dropout_rate)
            self.shortcut = None
            if not self.equal_in_out:
                self.shortcut = nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self.equal_in_out:
                x = self.relu1(self.bn1(x))
                out = x
            else:
                out = self.relu1(self.bn1(x))
            out = self.conv1(out)
            out = self.relu2(self.bn2(out))
            if self.dropout_rate > 0.0:
                out = F.dropout(out, p=self.dropout_rate, training=self.training)
            out = self.conv2(out)
            residual = x if self.shortcut is None else self.shortcut(x)
            return residual + out

    class NetworkBlock(nn.Module):
        def __init__(self, n_layers: int, in_planes: int, out_planes: int, stride: int) -> None:
            super().__init__()
            layers = []
            for i in range(n_layers):
                layers.append(
                    BasicBlock(
                        in_planes if i == 0 else out_planes,
                        out_planes,
                        stride if i == 0 else 1,
                    )
                )
            self.layer = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer(x)

    class WideResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if (depth - 4) % 6 != 0:
                raise ValueError(f"depth must satisfy (depth - 4) divisible by 6, got {depth}")
            n_layers = (depth - 4) // 6
            channels = [16, 16 * widen, 32 * widen, 64 * widen]
            self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False)
            self.block1 = NetworkBlock(n_layers, channels[0], channels[1], 1)
            self.block2 = NetworkBlock(n_layers, channels[1], channels[2], 2)
            self.block3 = NetworkBlock(n_layers, channels[2], channels[3], 2)
            self.bn1 = nn.BatchNorm2d(channels[3])
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(channels[3], num_classes)
            self.channels = channels[3]
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, nn.Linear):
                    nn.init.constant_(module.bias, 0.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.block3(self.block2(self.block1(self.conv1(x))))
            out = self.relu(self.bn1(out))
            out = F.adaptive_avg_pool2d(out, 1).view(-1, self.channels)
            return self.fc(out)

    return WideResNet()


def train_classifier(loader: Iterable, seed: int, epochs: int | None = None, device: str | None = None):
    """SGD with Nesterov momentum and a cosine learning-rate schedule."""
    import torch
    import torch.nn as nn

    cfg = SETTINGS["classifier"]
    epochs = int(cfg["epochs"] if epochs is None else epochs)
    set_seed(seed)
    device = device or preferred_device()
    model = build_wide_resnet().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(images), labels).backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    return model


def class_probability(model, loader: Iterable) -> np.ndarray:
    """Probability of the true class for each example, in loader order."""
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            probability = F.softmax(model(images), dim=1)
            chosen = probability.gather(1, labels.view(-1, 1)).squeeze(1)
            values.append(chosen.detach().cpu().numpy())
    if not values:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(values).astype(np.float64)


def classifier_scores(victim: np.ndarray, reference_rows: np.ndarray) -> dict[str, np.ndarray]:
    """``victim`` is P(true class). ``reference_rows`` has one reference model per row."""
    victim = np.asarray(victim, dtype=np.float64)
    return {
        "logprob": np.log(np.clip(victim, 1e-12, 1.0)),
        "ratio": likelihood_ratio(victim, reference_rows),
    }
