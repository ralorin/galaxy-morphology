"""The models we compare, plus the two wrappers used in the ablations.

Three groups:

*   `cnn_small`, `cnn_deep`, `cnn_deep_reg` are the plain convolutional networks
    from our earlier comparison, ported to PyTorch layer for layer so that the old
    baselines stay in the picture. They train from scratch.

*   Five convolutional backbones (ResNet-50, DenseNet-121, EfficientNetV2-S,
    ConvNeXt-Tiny, MobileNetV3-Large) and four transformers (ViT-S/16, ViT-B/16,
    DeiT3-S/16, Swin-T) come from `timm` with ImageNet weights.

*   `OrientationPooled` turns any of the above into a network that is invariant to
    the dihedral group by construction: it averages the logit over the eight
    rotations and reflections of the input. It costs eight forward passes, which
    is the honest price of the invariance, and it is what the `d4` augmentation
    only approximates.

`python -m src.models --download` builds every pretrained backbone once so the
weights land in the local cache. Compute nodes have no outbound network, so this
has to be done from the login node before submitting anything.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets import D4_ELEMENTS, Normalisation
from src.registry import ARCHITECTURES, FINETUNE_MODES, REGISTRY, Spec  # noqa: F401


# --------------------------------------------------------------------------- #
# The scratch baselines
# --------------------------------------------------------------------------- #

class SimpleCNN(nn.Module):
    """Conv-pool stack, flatten, one hidden layer, one logit.

    This is the shape of the networks in our earlier study: `channels` gives the
    convolutional widths, `hidden` the size of the dense layer. The L2 penalty
    those models applied per layer is handled by the optimiser's weight decay
    instead, which is the same objective written differently.
    """

    def __init__(self, channels=(16, 32, 64), hidden: int = 125,
                 input_size: int = 128, dropout: float = 0.0):
        super().__init__()
        blocks = []
        in_ch = 3
        spatial = input_size
        for out_ch in channels:
            blocks += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            in_ch = out_ch
            spatial //= 2
        self.features = nn.Sequential(*blocks)
        self.flat_dim = in_ch * spatial * spatial
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.flat_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


def _build_custom(name: str, input_size: int) -> nn.Module:
    if name == "cnn_small":
        return SimpleCNN((16, 32, 64), hidden=125, input_size=input_size)
    if name == "cnn_deep":
        return SimpleCNN((32, 64, 128, 256), hidden=256, input_size=input_size)
    if name == "cnn_deep_reg":
        # same shape as cnn_deep; the difference is a heavier weight decay and
        # dropout, i.e. the strongly regularised variant
        return SimpleCNN((32, 64, 128, 256), hidden=256, input_size=input_size, dropout=0.5)
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# Orientation pooling
# --------------------------------------------------------------------------- #

class OrientationPooled(nn.Module):
    """Average the logit over the eight elements of D4.

    Exactly invariant: rotating or mirroring the input permutes the eight branches
    and leaves the mean untouched. We pool the logit rather than the probability
    because averaging in logit space is the geometric mean of the odds, which is
    better behaved when one branch is very confident.
    """

    def __init__(self, model: nn.Module, reduction: str = "mean"):
        super().__init__()
        self.model = model
        if reduction not in ("mean", "max"):
            raise ValueError(reduction)
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = []
        for k, mirror in D4_ELEMENTS:
            view = torch.rot90(x, k=k, dims=(2, 3))
            if mirror:
                view = torch.flip(view, dims=(3,))
            logits.append(self.model(view))
        stacked = torch.stack(logits, dim=0)
        return stacked.mean(0) if self.reduction == "mean" else stacked.amax(0)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def build(name: str, input_size: int | None = None, pretrained: bool = True,
          finetune: str = "full", orientation_pooled: bool = False,
          drop_rate: float = 0.0) -> tuple[nn.Module, Normalisation, int]:
    """Return (model, normalisation, effective input size).

    The model emits a single logit, so every loss in `train.py` is a form of
    binary cross-entropy and nothing depends on the architecture.
    """
    if name not in REGISTRY:
        raise SystemExit(f"unknown architecture {name!r}; known: {', '.join(ARCHITECTURES)}")
    spec = REGISTRY[name]
    size = spec.input_size if input_size is None else int(input_size)
    if spec.fixed_size and size != spec.input_size:
        raise SystemExit(f"{name} only runs at {spec.input_size} px "
                         f"(its window size fixes the feature-map geometry)")

    if spec.family == "custom":
        model = _build_custom(name, size)
        norm = Normalisation(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    else:
        import timm

        kwargs = dict(pretrained=pretrained, num_classes=1, drop_rate=drop_rate)
        if size != spec.input_size and not spec.fixed_size:
            # timm interpolates the position embeddings for transformers and simply
            # accepts the new size for the convnets
            kwargs["img_size"] = size if spec.family == "transformer" else None
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
        model = timm.create_model(spec.timm_id, **kwargs)
        cfg = model.pretrained_cfg or {}
        norm = Normalisation(mean=tuple(cfg.get("mean", (0.485, 0.456, 0.406))),
                             std=tuple(cfg.get("std", (0.229, 0.224, 0.225))))
        _apply_finetune(model, finetune)
        model = _SqueezeLogit(model)

    if orientation_pooled:
        model = OrientationPooled(model)
    return model, norm, size


class _SqueezeLogit(nn.Module):
    """timm heads give (B, 1); the losses want (B,)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


def _apply_finetune(model: nn.Module, mode: str) -> None:
    """Freeze the backbone according to `mode`.

    frozen   only the classifier head trains (the linear-probe setting)
    partial  head plus the last stage or the last two transformer blocks
    full     everything trains
    """
    if mode not in FINETUNE_MODES:
        raise SystemExit(f"unknown finetune mode {mode!r}")
    if mode == "full":
        return

    for p in model.parameters():
        p.requires_grad = False

    head = model.get_classifier() if hasattr(model, "get_classifier") else None
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True

    if mode == "partial":
        for group in _last_stage(model):
            for p in group.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  finetune={mode}: {trainable:,} / {total:,} parameters trainable")


def _last_stage(model: nn.Module) -> list[nn.Module]:
    """Best effort at "the last stage", across the families we use."""
    # transformers: the final blocks and the norm before the head
    if hasattr(model, "blocks"):
        blocks = list(model.blocks)
        out = blocks[-2:]
        if hasattr(model, "norm"):
            out.append(model.norm)
        return out
    # swin and convnext expose stages
    if hasattr(model, "layers"):
        return [list(model.layers)[-1]]
    if hasattr(model, "stages"):
        return [list(model.stages)[-1]]
    # resnet
    if hasattr(model, "layer4"):
        return [model.layer4]
    # densenet / efficientnet / mobilenet keep everything in `features`
    if hasattr(model, "features"):
        children = list(model.features.children())
        return children[-2:]
    return []


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params": int(total), "params_trainable": int(trainable)}


def estimate_flops(model: nn.Module, input_size: int) -> float | None:
    """GFLOPs for one forward pass, if fvcore happens to be installed.

    Not a dependency: the number is nice to have in the efficiency table but the
    pipeline does not need it.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    model = model.eval()
    x = torch.zeros(1, 3, input_size, input_size)
    try:
        counter = FlopCountAnalysis(model, x)
        counter.unsupported_ops_warnings(False)
        counter.uncalled_modules_warnings(False)
        return float(counter.total()) / 1e9
    except Exception:
        return None


def d4_invariance_error(model: nn.Module, images: torch.Tensor) -> float:
    """Mean absolute spread of the predicted probability over the eight D4 views.

    Zero for an `OrientationPooled` model by construction; for everything else it
    says how much of the group the network failed to learn. Used in the
    orientation ablation as a direct measurement rather than a proxy.
    """
    model.eval()
    with torch.no_grad():
        probs = []
        for k, mirror in D4_ELEMENTS:
            view = torch.rot90(images, k=k, dims=(2, 3))
            if mirror:
                view = torch.flip(view, dims=(3,))
            probs.append(torch.sigmoid(model(view)))
        stacked = torch.stack(probs, dim=0)
    return float((stacked.max(0).values - stacked.min(0).values).mean())


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true",
                    help="build every pretrained backbone once to warm the weight cache")
    ap.add_argument("--summary", action="store_true",
                    help="print parameter counts for the whole registry")
    args = ap.parse_args()

    if args.download:
        for name, spec in REGISTRY.items():
            if spec.timm_id is None:
                continue
            print(f"fetching {spec.timm_id}")
            build(name, pretrained=True)
        print("weight cache warmed")

    if args.summary or not args.download:
        print(f"{'model':16s} {'family':12s} {'size':>5s} {'params':>14s} {'GFLOPs':>8s}")
        for name, spec in REGISTRY.items():
            model, _, size = build(name, pretrained=False)
            counts = count_parameters(model)
            flops = estimate_flops(model, size)
            flop_str = f"{flops:8.2f}" if flops else "       -"
            print(f"{name:16s} {spec.family:12s} {size:5d} "
                  f"{counts['params']:14,d} {flop_str}")


if __name__ == "__main__":
    main()
