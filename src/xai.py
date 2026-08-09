"""Saliency maps and, more importantly, numbers that say whether they mean anything.

    python -m src.xai --runs resnet50_soft_d4_s224_full_bce_nfull_seed0 --n 12
    python -m src.xai --all-checkpoints          # every run that saved weights

For each run this produces

    results/xai/<run_id>.json    faithfulness scores averaged over the sample
    results/xai/<run_id>.csv     per-galaxy scores
    figures/xai_<run_id>.png     an overlay panel for the qualitative figure

Methods. Convolutional networks get Grad-CAM and Grad-CAM++ on the last spatial
stage. Transformers get attention rollout, which is the closest equivalent that
does not depend on a spatial feature map. Swin is the awkward case: its shifted
windows make rollout ill-defined, so it is treated like a convnet on the output of
its last stage.

Why the numbers matter. A heatmap that lands on the galaxy looks convincing whether
or not the network used that region, so we score the maps three ways instead of
eyeballing them:

*   deletion AUC   blank out the most salient pixels first and watch the predicted
                   probability fall. A faithful map makes it fall fast, so lower is
                   better.
*   insertion AUC  start from a blurred image and restore the most salient pixels
                   first. Higher is better.
*   background reliance
                   the share of the attribution mass that falls outside the
                   galaxy's own footprint, where by construction there is nothing
                   relevant. This one is specific to this problem and it is the
                   one that catches a network keying on sky noise or on a
                   neighbour at the edge of the crop.

The footprint comes from the image itself: threshold at the sky level estimated
from the border, keep the connected component that contains the centre. That is a
crude segmentation but it does not need to be precise, only unbiased with respect
to the model being scored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from src import models
from src.common import ensure_dir, get_device, read_json, set_seed, write_json
from src.datasets import GalaxyDataset, load_gz2_table


# --------------------------------------------------------------------------- #
# Where to hook
# --------------------------------------------------------------------------- #

def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip the logit-squeeze and orientation-pooling wrappers."""
    inner = model
    while hasattr(inner, "model") and not hasattr(inner, "blocks"):
        inner = inner.model
    return inner


def target_layer(arch: str, model: torch.nn.Module) -> torch.nn.Module:
    inner = unwrap(model)
    if arch.startswith("cnn_"):
        convs = [m for m in inner.features if isinstance(m, torch.nn.Conv2d)]
        return convs[-1]
    if arch == "resnet50":
        return inner.layer4
    if arch == "densenet121":
        return inner.features
    if arch == "effnetv2_s":
        return inner.blocks[-1]
    if arch == "convnext_tiny":
        return inner.stages[-1]
    if arch == "mobilenetv3":
        return inner.blocks[-1]
    if arch == "swin_tiny":
        return inner.layers[-1]
    raise SystemExit(f"no CAM target layer defined for {arch}")


# --------------------------------------------------------------------------- #
# Grad-CAM and Grad-CAM++
# --------------------------------------------------------------------------- #

class GradCAM:
    """Grad-CAM (Selvaraju et al.) and its ++ variant, on one target layer.

    Both are implemented here rather than pulled in from a library because the
    only fiddly part is the layout of the captured activations, which differs
    between the families we use, and it is easier to see that in one place.
    """

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module, plus: bool = False):
        self.model = model
        self.plus = plus
        self.activations = None
        self.gradients = None
        self._handles = [
            layer.register_forward_hook(self._save_activation),
            layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, out):
        self.activations = self._to_nchw(out)

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = self._to_nchw(grad_out[0])

    @staticmethod
    def _to_nchw(t: torch.Tensor) -> torch.Tensor:
        # Swin and recent ConvNeXt stages emit (B, H, W, C); everything else is
        # already channels-first.
        if t.dim() == 4 and t.shape[1] == t.shape[2] and t.shape[3] != t.shape[1]:
            return t.permute(0, 3, 1, 2)
        return t

    def close(self) -> None:
        for h in self._handles:
            h.remove()

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (saliency (B, H, W) in [0,1], logits (B,))."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits.sum().backward()

        a, g = self.activations, self.gradients
        if a is None or g is None:
            raise RuntimeError("hooks did not fire; check the target layer")

        if not self.plus:
            weights = g.mean(dim=(2, 3), keepdim=True)
        else:
            g2, g3 = g ** 2, g ** 3
            denom = 2 * g2 + (a * g3).sum(dim=(2, 3), keepdim=True)
            alpha = g2 / denom.clamp_min(1e-8)
            weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * a).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear",
                            align_corners=False).squeeze(1)
        return _normalise(cam.detach()), logits.detach()


def _normalise(cam: torch.Tensor) -> torch.Tensor:
    flat = cam.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1)
    return (cam - lo) / (hi - lo).clamp_min(1e-8)


# --------------------------------------------------------------------------- #
# Attention rollout
# --------------------------------------------------------------------------- #

class AttentionRollout:
    """Abnar and Zuidema's rollout: multiply the head-averaged attention matrices,
    with the residual stream folded in as an identity term.

    timm's attention blocks use the fused kernel by default, which never
    materialises the attention matrix, so we switch that off for the duration.
    """

    def __init__(self, model: torch.nn.Module, discard_ratio: float = 0.9):
        self.inner = unwrap(model)
        self.model = model
        self.discard_ratio = discard_ratio
        self.maps: list[torch.Tensor] = []
        self._handles = []
        self._was_fused = []

        for block in self.inner.blocks:
            attn = block.attn
            self._was_fused.append(getattr(attn, "fused_attn", False))
            if hasattr(attn, "fused_attn"):
                attn.fused_attn = False
            self._handles.append(attn.attn_drop.register_forward_hook(self._save))

    def _save(self, _module, inp, _out):
        self.maps.append(inp[0].detach())

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        for block, fused in zip(self.inner.blocks, self._was_fused):
            if hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = fused

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.maps.clear()
        logits = self.model(x)
        if not self.maps:
            raise RuntimeError("no attention captured; the block layout changed")

        b, tokens = self.maps[0].shape[0], self.maps[0].shape[-1]
        result = torch.eye(tokens, device=x.device).expand(b, tokens, tokens).clone()
        for attn in self.maps:
            a = attn.mean(dim=1)  # average the heads
            if self.discard_ratio > 0:
                # drop the weakest links, as in the original implementation: they
                # are mostly noise and they wash the product out
                flat = a.flatten(1)
                k = int(flat.shape[1] * self.discard_ratio)
                if k > 0:
                    threshold = flat.kthvalue(k, dim=1).values.view(-1, 1, 1)
                    a = torch.where(a < threshold, torch.zeros_like(a), a)
            a = a + torch.eye(tokens, device=x.device)
            a = a / a.sum(dim=-1, keepdim=True)
            result = a @ result

        n_prefix = tokens - (x.shape[-1] // self._patch_size()) ** 2
        cls_to_patches = result[:, 0, n_prefix:]
        side = int(round(cls_to_patches.shape[-1] ** 0.5))
        cam = cls_to_patches.reshape(b, side, side)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear",
                            align_corners=False).squeeze(1)
        return _normalise(cam), logits.detach()

    def _patch_size(self) -> int:
        pe = getattr(self.inner, "patch_embed", None)
        size = getattr(pe, "patch_size", (16, 16))
        return int(size[0] if isinstance(size, (tuple, list)) else size)


def make_explainer(arch: str, model: torch.nn.Module, method: str):
    family = models.REGISTRY[arch].family
    if method == "auto":
        method = "rollout" if (family == "transformer" and arch != "swin_tiny") else "gradcam"
    if method == "rollout":
        return AttentionRollout(model), method
    return GradCAM(model, target_layer(arch, model), plus=(method == "gradcam++")), method


# --------------------------------------------------------------------------- #
# Faithfulness
# --------------------------------------------------------------------------- #

@torch.no_grad()
def deletion_insertion(model, x: torch.Tensor, cam: torch.Tensor,
                       steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Curves of the predicted probability as pixels are removed or restored.

    Deletion replaces the most salient pixels with the per-image mean, which is the
    closest thing to "no information" for an image that is mostly sky. Insertion
    starts from a heavily blurred copy and restores detail in order of salience.
    """
    b = x.shape[0]
    n_pixels = x.shape[-1] * x.shape[-2]
    order = cam.flatten(1).argsort(dim=1, descending=True)

    baseline = x.mean(dim=(2, 3), keepdim=True).expand_as(x).clone()
    blurred = _gaussian_blur(x, sigma=8.0)

    del_curve, ins_curve = [], []
    for step in range(steps + 1):
        k = int(round(n_pixels * step / steps))
        mask = torch.zeros(b, n_pixels, device=x.device)
        if k:
            mask.scatter_(1, order[:, :k], 1.0)
        mask = mask.view(b, 1, *x.shape[-2:])

        deleted = x * (1 - mask) + baseline * mask
        inserted = blurred * (1 - mask) + x * mask
        del_curve.append(torch.sigmoid(model(deleted)).float().cpu().numpy())
        ins_curve.append(torch.sigmoid(model(inserted)).float().cpu().numpy())

    return np.stack(del_curve, axis=1), np.stack(ins_curve, axis=1)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = int(3 * sigma)
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    c = x.shape[1]
    kx = kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    out = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kx, groups=c)
    out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="reflect"), ky, groups=c)
    return out


def galaxy_footprint(image_uint8: np.ndarray, border: int = 12) -> np.ndarray:
    """Boolean mask of the central source.

    Sky level and noise are estimated from a frame of `border` pixels around the
    edge, which is background by construction after the central crop. Everything
    above sky + 3 sigma is a source; we keep the connected component that touches
    the centre and dilate it slightly so that faint outer arms are inside.
    """
    import cv2

    grey = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    frame = np.concatenate([grey[:border].ravel(), grey[-border:].ravel(),
                            grey[:, :border].ravel(), grey[:, -border:].ravel()])
    sky = float(np.median(frame))
    # median absolute deviation, scaled to a Gaussian sigma
    noise = 1.4826 * float(np.median(np.abs(frame - sky))) + 1e-6
    mask = (grey > sky + 3.0 * noise).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, labels = cv2.connectedComponents(mask)
    h, w = grey.shape
    centre_label = labels[h // 2, w // 2]
    if centre_label == 0:
        # nothing detected at the centre: fall back to a central disc so that the
        # metric stays defined instead of silently becoming zero
        yy, xx = np.ogrid[:h, :w]
        return ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) <= (0.3 * h) ** 2
    keep = (labels == centre_label).astype(np.uint8)
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8), iterations=2)
    return keep.astype(bool)


def background_reliance(cam: np.ndarray, footprint: np.ndarray) -> float:
    total = float(cam.sum())
    if total <= 0:
        return float("nan")
    return float(cam[~footprint].sum() / total)


# --------------------------------------------------------------------------- #
# Driving it
# --------------------------------------------------------------------------- #

def disable_inplace(model: torch.nn.Module) -> None:
    """Turn off every in-place activation in the network.

    `register_full_backward_hook` wraps a module's output in a custom autograd
    function, and an in-place ReLU applied to that wrapped output is a view being
    modified in place, which autograd refuses. Since most of our backbones use
    `ReLU(inplace=True)` this would break Grad-CAM on almost all of them. Switching
    the flag off costs a little activation memory and changes nothing numerically:
    the forward is the same function either way, and we are not training here.
    """
    for module in model.modules():
        if getattr(module, "inplace", False):
            module.inplace = False


def load_run(run_id: str):
    out_dir = config.RUNS / run_id
    ckpt = out_dir / "checkpoint.pt"
    if not ckpt.exists():
        raise SystemExit(f"{run_id} has no checkpoint.pt")
    cfg = read_json(out_dir / "metrics.json")["config"]
    model, norm, size = models.build(cfg["arch"], input_size=cfg["size"], pretrained=False,
                                     finetune="full",
                                     orientation_pooled=cfg["orientation_pooled"])
    # we wrote these checkpoints ourselves, so the pickle is ours to trust
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state)
    disable_inplace(model)
    return model.eval(), norm, cfg


def sample_test_galaxies(n: int, seed: int = 0) -> pd.DataFrame:
    """A stratified sample across the agreement range, so the panel shows the easy
    and the contested cases side by side rather than n unanimous ellipticals."""
    table = load_gz2_table()
    test = table[table["split"] == "test"]
    rng = np.random.default_rng(seed)
    edges = np.array(config.AGREEMENT_BINS)
    per_bin = max(1, n // (len(edges) - 1))
    parts = []
    for b in range(len(edges) - 1):
        sub = test[(test["agreement"] >= edges[b]) & (test["agreement"] < edges[b + 1] + 1e-9)]
        if sub.empty:
            continue
        parts.append(sub.iloc[rng.permutation(len(sub))[:per_bin]])
    return pd.concat(parts).reset_index(drop=True)


def explain_run(run_id: str, n: int = 24, method: str = "auto", steps: int = 20,
                make_panel: bool = True) -> dict:
    set_seed(config.SEED, deterministic=True)
    device = get_device()
    model, norm, cfg = load_run(run_id)
    model = model.to(device)

    sample = sample_test_galaxies(n)
    ds = GalaxyDataset(sample, config.ARRAYS / "gz2_images.npy", cfg["size"], "none",
                       norm, train=False)
    raw_images = np.load(config.ARRAYS / "gz2_images.npy", mmap_mode="r")

    explainer, used = make_explainer(cfg["arch"], model, method)
    print(f"{run_id}: {used} on {len(ds)} galaxies")

    rows, cams = [], []
    try:
        for i in range(len(ds)):
            item = ds[i]
            x = item["image"].unsqueeze(0).to(device)
            cam, logits = explainer(x)
            prob = float(torch.sigmoid(logits).item())

            raw = np.asarray(raw_images[int(item["row"])])
            if raw.shape[0] != cfg["size"]:
                import cv2
                raw = cv2.resize(raw, (cfg["size"], cfg["size"]),
                                 interpolation=cv2.INTER_AREA)
            footprint = galaxy_footprint(raw)
            cam_np = cam[0].cpu().numpy()
            cams.append((raw, cam_np))

            del_curve, ins_curve = deletion_insertion(model, x, cam, steps)
            rows.append({
                "run_id": run_id,
                "row": int(item["row"]),
                "label": int(item["label"]),
                "p_featured": float(item["target"]),
                "agreement": float(item["agreement"]),
                "prob": prob,
                "correct": int((prob >= 0.5) == bool(item["label"])),
                "deletion_auc": float(del_curve[0].mean()),
                "insertion_auc": float(ins_curve[0].mean()),
                "background_reliance": background_reliance(cam_np, footprint),
                "footprint_fraction": float(footprint.mean()),
            })
    finally:
        explainer.close()

    per_galaxy = pd.DataFrame(rows)
    out_dir = ensure_dir(config.RESULTS / "xai")
    per_galaxy.to_csv(out_dir / f"{run_id}.csv", index=False)

    summary = {
        "run_id": run_id,
        "arch": cfg["arch"],
        "label_mode": cfg["label_mode"],
        "method": used,
        "n": int(len(per_galaxy)),
        "deletion_auc": float(per_galaxy["deletion_auc"].mean()),
        "insertion_auc": float(per_galaxy["insertion_auc"].mean()),
        "background_reliance": float(per_galaxy["background_reliance"].mean()),
        # the same three, restricted to the galaxies the volunteers agreed on
        "background_reliance_high_agreement": float(
            per_galaxy.loc[per_galaxy["agreement"] >= 0.6, "background_reliance"].mean()),
        "background_reliance_low_agreement": float(
            per_galaxy.loc[per_galaxy["agreement"] < 0.6, "background_reliance"].mean()),
        "accuracy_on_sample": float(per_galaxy["correct"].mean()),
    }
    write_json(out_dir / f"{run_id}.json", summary)

    if make_panel:
        _panel(run_id, cams, per_galaxy)
    print(f"  deletion {summary['deletion_auc']:.3f}  insertion {summary['insertion_auc']:.3f}  "
          f"background {summary['background_reliance']:.3f}")
    return summary


def _panel(run_id: str, cams, per_galaxy: pd.DataFrame, cols: int = 6) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(cams), 2 * cols)
    order = per_galaxy["agreement"].to_numpy().argsort()[::-1][:n]
    fig, axes = plt.subplots(2, cols, figsize=(2.1 * cols, 4.6))
    for ax, idx in zip(axes.ravel(), order):
        raw, cam = cams[idx]
        row = per_galaxy.iloc[idx]
        ax.imshow(raw)
        ax.imshow(cam, cmap="inferno", alpha=0.45)
        ax.set_title(f"a={row['agreement']:.2f}  p={row['prob']:.2f}", fontsize=7)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(run_id, fontsize=8)
    fig.tight_layout()
    out = ensure_dir(config.FIGURES) / f"xai_{run_id}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--all-checkpoints", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--method", choices=["auto", "gradcam", "gradcam++", "rollout"],
                    default="auto")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    run_ids = args.runs or []
    if args.all_checkpoints:
        run_ids = sorted(p.parent.name for p in config.RUNS.glob("*/checkpoint.pt"))
    if not run_ids:
        raise SystemExit("nothing to do: pass --runs or --all-checkpoints")

    summaries = []
    for run_id in run_ids:
        try:
            summaries.append(explain_run(run_id, args.n, args.method, args.steps))
        except SystemExit as exc:
            print(f"skipping {run_id}: {exc}")

    if summaries:
        out = pd.DataFrame(summaries)
        out.to_csv(config.RESULTS / "xai_summary.csv", index=False)
        print(f"\nwrote {config.RESULTS / 'xai_summary.csv'}")
        print(out[["arch", "label_mode", "method", "deletion_auc", "insertion_auc",
                   "background_reliance"]].to_string(index=False))


if __name__ == "__main__":
    main()
