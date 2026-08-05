"""What a run is: its settings, its defaults and its identifier.

Separated from `train.py` so that `build_jobs.py` and the analysis scripts can
reason about configurations without importing torch. The one rule to keep in mind
is that `run_id` must stay a pure function of the resolved configuration: it is the
name of the output directory, so it is also how a resubmitted array task discovers
that its work is already done.
"""

from __future__ import annotations

from src.registry import REGISTRY

LABEL_MODES = ("hard", "hard_conf", "soft", "soft_w")
LOSSES = ("bce", "focal", "smooth")
POLICIES = ("none", "flip", "d4", "d4_photo")

# |2p-1| >= 0.6  <=>  p <= 0.2 or p >= 0.8
CONFIDENT_AGREEMENT = 0.6


def default_config() -> dict:
    return {
        "arch": "resnet50",
        "label_mode": "hard",
        "policy": "d4",
        "size": None,            # None -> the architecture's native size
        "finetune": "full",
        "loss": "bce",
        "train_size": None,      # None -> the whole training split
        "seed": 0,
        "orientation_pooled": False,
        "pretrained": True,      # ImageNet weights vs random initialisation
        "epochs": 20,
        "batch_size": 128,
        "lr": None,              # None -> chosen by family in resolve()
        "weight_decay": None,
        "patience": 5,
        "warmup_epochs": 1,
        "tta": True,             # also report D4 test-time averaging
        "save_checkpoint": False,
    }


def run_id(cfg: dict) -> str:
    parts = [
        cfg["arch"], cfg["label_mode"], cfg["policy"], f"s{cfg['size']}",
        cfg["finetune"], cfg["loss"],
        f"n{cfg['train_size']}" if cfg["train_size"] else "nfull",
        f"seed{cfg['seed']}",
    ]
    if cfg["orientation_pooled"]:
        parts.append("op")
    if not cfg.get("pretrained", True):
        parts.append("scratch")
    return "_".join(str(p) for p in parts)


def resolve(cfg: dict) -> dict:
    """Fill in the settings we did not want to repeat in every job row."""
    keep_none = ("size", "train_size", "lr", "weight_decay")
    merged = {**default_config(),
              **{k: v for k, v in cfg.items() if v is not None or k in keep_none}}
    spec = REGISTRY[merged["arch"]]
    if merged["size"] is None:
        merged["size"] = spec.input_size

    if merged["lr"] is None:
        # scratch nets want a much larger step than a fine-tuned backbone, and
        # transformers are the most sensitive of the three
        merged["lr"] = {"custom": 1e-3, "cnn": 3e-4, "transformer": 1e-4}[spec.family]
        if merged["finetune"] == "frozen":
            merged["lr"] *= 10   # only a linear head is moving
        elif not merged["pretrained"]:
            merged["lr"] *= 3    # nothing to preserve, so a larger step converges sooner
    if merged["weight_decay"] is None:
        merged["weight_decay"] = 0.05 if merged["arch"] == "cnn_deep_reg" else 0.01

    if merged["orientation_pooled"]:
        # eight forward passes per step; shrink the batch to stay inside 80 GB
        merged["batch_size"] = max(16, merged["batch_size"] // 4)
    if merged["size"] > 224:
        merged["batch_size"] = max(16, merged["batch_size"] // 2)
    return merged
