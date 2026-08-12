"""What each model is, without importing torch.

`models.py` needs this to build networks, but so do the analysis, table and figure
scripts, which have no business pulling in torch and timm just to know that
`vit_small` is a transformer that runs at 224 px. Keeping the metadata here means
the whole post-processing half of the pipeline runs on a login node or a laptop
with nothing but pandas installed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    name: str
    family: str            # "custom" | "cnn" | "transformer"
    timm_id: str | None
    input_size: int
    fixed_size: bool = False   # True when the architecture cannot change resolution
    pretty: str = ""


REGISTRY: dict[str, Spec] = {
    # our own baselines, trained from scratch
    "cnn_small":     Spec("cnn_small", "custom", None, 128, pretty="CNN-small"),
    "cnn_deep":      Spec("cnn_deep", "custom", None, 128, pretty="CNN-deep"),
    "cnn_deep_reg":  Spec("cnn_deep_reg", "custom", None, 128, pretty="CNN-deep+reg"),
    # convolutional backbones
    "resnet50":      Spec("resnet50", "cnn", "resnet50.a1_in1k", 224,
                          pretty="ResNet-50"),
    "densenet121":   Spec("densenet121", "cnn", "densenet121.ra_in1k", 224,
                          pretty="DenseNet-121"),
    "effnetv2_s":    Spec("effnetv2_s", "cnn", "tf_efficientnetv2_s.in1k", 224,
                          pretty="EfficientNetV2-S"),
    "convnext_tiny": Spec("convnext_tiny", "cnn", "convnext_tiny.fb_in1k", 224,
                          pretty="ConvNeXt-T"),
    "mobilenetv3":   Spec("mobilenetv3", "cnn", "mobilenetv3_large_100.ra_in1k", 224,
                          pretty="MobileNetV3-L"),
    # transformers
    "vit_small":     Spec("vit_small", "transformer",
                          "vit_small_patch16_224.augreg_in21k_ft_in1k", 224,
                          pretty="ViT-S/16"),
    "vit_base":      Spec("vit_base", "transformer",
                          "vit_base_patch16_224.augreg2_in21k_ft_in1k", 224,
                          pretty="ViT-B/16"),
    "deit3_small":   Spec("deit3_small", "transformer",
                          "deit3_small_patch16_224.fb_in1k", 224,
                          pretty="DeiT3-S/16"),
    # Swin's window size fixes the feature-map geometry, so it only runs at 224
    "swin_tiny":     Spec("swin_tiny", "transformer",
                          "swin_tiny_patch4_window7_224.ms_in1k", 224,
                          fixed_size=True, pretty="Swin-T"),
}

ARCHITECTURES = tuple(REGISTRY)
FINETUNE_MODES = ("frozen", "partial", "full")
FAMILIES = ("custom", "cnn", "transformer")

PRETTY_ARCH = {name: spec.pretty or name for name, spec in REGISTRY.items()}
PRETTY_FAMILY = {"custom": "Scratch CNN", "cnn": "Pretrained CNN",
                 "transformer": "Transformer"}


def reference_runs(runs, label_mode: str | None = None):
    """The one protocol every headline number comes from.

    $D_4$ augmentation, full fine-tuning from ImageNet weights, cross-entropy, the
    whole training split, no orientation pooling, and each architecture at its own
    native input resolution.

    That last condition is the one that is easy to forget and expensive to get wrong.
    The resolution sweep reuses the reference protocol and changes only the input
    size, so its runs satisfy every other condition and land in the pool unless the
    size is pinned. Four of the twelve architectures then average nine runs at three
    resolutions into a row the caption calls five seeds. Every module that needs the
    reference protocol calls this rather than restating the predicate, because the
    predicate had already drifted apart in five places.
    """
    native = runs["arch"].map(lambda a: REGISTRY[a].input_size if a in REGISTRY else None)
    keep = ((runs["policy"] == "d4")
            & (runs["finetune"] == "full")
            & (runs["loss"] == "bce")
            & (runs["train_size"].fillna(0).astype(int) == 0)
            & (~runs["orientation_pooled"].fillna(False).astype(bool))
            & (runs["pretrained"].fillna(True).astype(bool))
            & (runs["size"] == native))
    if label_mode is not None:
        keep &= runs["label_mode"] == label_mode
    return runs[keep]


def family_of(arch: str) -> str:
    spec = REGISTRY.get(arch)
    return spec.family if spec else "unknown"
