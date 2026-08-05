"""Train and evaluate one configuration.

Either give the settings on the command line

    python -m src.train --arch resnet50 --label-mode soft --policy d4 --seed 0

or point it at a row of the job table that `build_jobs.py` writes, which is how
the SLURM arrays call it

    python -m src.train --jobs-csv $GZM_WORK/jobs/jobs.csv --task-id $SLURM_ARRAY_TASK_ID

Everything a run produces goes into $GZM_WORK/runs/<run_id>/: the resolved
configuration, the epoch history, the test and validation probabilities, and the
metrics. A run whose metrics.json already exists is skipped unless --force is
given, so a job that hits the wall clock can simply be resubmitted.

Label modes
-----------
hard       target = 1[p > 0.5]; the usual setting, and what the literature does
hard_conf  same target, but the training set is restricted to galaxies the
           volunteers agreed on (agreement >= 0.6). Tests whether the ceiling is
           caused by noisy labels or by ambiguous images
soft       target = p, the raw vote fraction itself
soft_w     target = p, with each galaxy weighted by how many people looked at it
hard_debiased, soft_debiased
           the same two targets built from the redshift-debiased fractions of
           Hart et al. rather than the raw ones. They are the control for the
           choice of fraction, which matters here: on this catalogue the two
           thresholds disagree on about a third of the galaxies

The validation and test sets are never filtered, reweighted or relabelled: every
mode is scored against the same held-out galaxies with the same raw-fraction
labels, so the numbers are comparable.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from src import models
from src.common import (classification_metrics, describe_device, ensure_dir,
                        get_device, set_seed, write_json)
from src.experiment import (CONFIDENT_AGREEMENT, LABEL_MODES, LOSSES,
                            default_config, resolve, run_id)
from src.datasets import (GalaxyDataset, load_decals_table, load_gz2_table,
                          make_loader, subsample_train)

# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def compute_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
                 loss: str) -> torch.Tensor:
    """All three losses take soft targets, so the label mode and the loss are
    independent knobs."""
    if loss == "bce":
        per_sample = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    elif loss == "smooth":
        t = target * 0.9 + 0.05
        per_sample = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    elif loss == "focal":
        gamma = 2.0
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        per_sample = ((1 - pt) ** gamma) * ce
    else:
        raise SystemExit(f"unknown loss {loss!r}")
    return (per_sample * weight).sum() / weight.sum().clamp_min(1e-8)


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #

def build_targets(table: pd.DataFrame, label_mode: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply the label mode to a training table. Returns (table, weights).

    Only the training targets change here. The validation and test tables keep the
    raw-fraction label throughout, including for the two `*_debiased` modes, so
    that every mode is scored against the same held-out answers and the columns of
    the results table mean the same thing.
    """
    table = table.copy()

    if label_mode.endswith("_debiased"):
        if "p_featured_debiased" not in table:
            raise SystemExit("the table has no p_featured_debiased column; rerun "
                             "src.prepare_gz2 --stage table")
        table["p_featured"] = table["p_featured_debiased"].astype(np.float32)
        table["label"] = (table["p_featured"] > 0.5).astype(np.int8)
        print(f"  training on the debiased fractions "
              f"({100 * table['label'].mean():.1f}% featured in train)")

    if label_mode == "hard_conf":
        before = len(table)
        table = table[table["agreement"] >= CONFIDENT_AGREEMENT].reset_index(drop=True)
        print(f"  hard_conf keeps {len(table):,} of {before:,} training galaxies")

    if label_mode in ("hard", "hard_conf", "hard_debiased"):
        table["p_featured"] = table["label"].astype(np.float32)
        weights = np.ones(len(table), dtype=np.float32)
    elif label_mode in ("soft", "soft_debiased"):
        weights = np.ones(len(table), dtype=np.float32)
    elif label_mode == "soft_w":
        # more votes means a better measured fraction; cap the ratio so that a
        # handful of heavily classified galaxies cannot dominate a batch
        v = table["votes"].to_numpy(dtype=np.float32)
        weights = np.clip(v / np.median(v), 0.25, 4.0)
    else:
        raise SystemExit(f"unknown label mode {label_mode!r}")
    return table, weights


def build_datasets(cfg: dict, norm) -> tuple[GalaxyDataset, GalaxyDataset, GalaxyDataset]:
    table = load_gz2_table()
    images = config.ARRAYS / "gz2_images.npy"

    train = table[table["split"] == "train"].reset_index(drop=True)
    train = subsample_train(train, cfg["train_size"], cfg["seed"])
    train, weights = build_targets(train, cfg["label_mode"])

    val = table[table["split"] == "val"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)

    ds_train = GalaxyDataset(train, images, cfg["size"], cfg["policy"], norm,
                             train=True, weights=weights)
    ds_val = GalaxyDataset(val, images, cfg["size"], "none", norm, train=False)
    ds_test = GalaxyDataset(test, images, cfg["size"], "none", norm, train=False)
    print(f"  train {len(ds_train):,} | val {len(ds_val):,} | test {len(ds_test):,}")
    return ds_train, ds_val, ds_test


# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict(model, loader, device, amp_dtype, tta: bool = False) -> pd.DataFrame:
    """Probabilities for every item in `loader`.

    With `tta` the eight D4 views are averaged in logit space, matching what
    `OrientationPooled` does internally.
    """
    model.eval()
    rows, probs, probs_tta, labels, targets, agreement = [], [], [], [], [], []
    from src.datasets import D4_ELEMENTS

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logit = model(x).float()
            if tta:
                acc = logit.clone()
                for k, mirror in D4_ELEMENTS[1:]:
                    view = torch.rot90(x, k=k, dims=(2, 3))
                    if mirror:
                        view = torch.flip(view, dims=(3,))
                    acc += model(view).float()
                logit_tta = acc / len(D4_ELEMENTS)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        if tta:
            probs_tta.append(torch.sigmoid(logit_tta).cpu().numpy())
        rows.append(batch["row"].numpy())
        labels.append(batch["label"].numpy())
        targets.append(batch["target"].numpy())
        agreement.append(batch["agreement"].numpy())

    out = pd.DataFrame({
        "row": np.concatenate(rows),
        "label": np.concatenate(labels),
        "p_featured": np.concatenate(targets),
        "agreement": np.concatenate(agreement),
        "prob": np.concatenate(probs).astype(np.float64),
    })
    if tta:
        out["prob_tta"] = np.concatenate(probs_tta).astype(np.float64)
    return out


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #

def train_one(cfg: dict, workers: int = 8, force: bool = False) -> dict:
    cfg = resolve(cfg)
    rid = run_id(cfg)
    out_dir = ensure_dir(config.RUNS / rid)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not force:
        print(f"{rid}: already done, skipping")
        from src.common import read_json
        return read_json(metrics_path)

    print(f"\n=== {rid} ===")
    print(f"  device: {describe_device()}")
    set_seed(cfg["seed"])
    device = get_device()
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if not cfg["pretrained"] and cfg["finetune"] != "full":
        raise SystemExit("a randomly initialised backbone with a frozen body would "
                         "train a head on random features; use finetune=full")
    model, norm, size = models.build(
        cfg["arch"], input_size=cfg["size"], pretrained=cfg["pretrained"],
        finetune=cfg["finetune"], orientation_pooled=cfg["orientation_pooled"])
    cfg["size"] = size
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    ds_train, ds_val, ds_test = build_datasets(cfg, norm)
    dl_train = make_loader(ds_train, cfg["batch_size"], True, workers, drop_last=True)
    dl_val = make_loader(ds_val, cfg["batch_size"] * 2, False, workers)
    dl_test = make_loader(ds_test, cfg["batch_size"] * 2, False, workers)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = max(1, len(dl_train))
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = steps_per_epoch * cfg["warmup_epochs"]

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    history = []
    best = {"auc": -1.0, "epoch": -1, "state": None}
    step = 0
    train_start = time.perf_counter()

    for epoch in range(cfg["epochs"]):
        model.train()
        running, seen, t0 = 0.0, 0, time.perf_counter()
        for batch in dl_train:
            x = batch["image"].to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            target = batch["target"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(x)
                loss = compute_loss(logits.float(), target, weight, cfg["loss"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            step += 1
            running += loss.item() * x.size(0)
            seen += x.size(0)

        val_pred = predict(model, dl_val, device, amp_dtype, tta=False)
        val_metrics = classification_metrics(val_pred["label"], val_pred["prob"])
        history.append({
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "lr": opt.param_groups[0]["lr"],
            "seconds": time.perf_counter() - t0,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        print(f"  epoch {epoch:2d}  loss {history[-1]['train_loss']:.4f}  "
              f"val acc {val_metrics['accuracy']:.4f}  val auc {val_metrics['auc_roc']:.4f}  "
              f"({history[-1]['seconds']:.0f} s)", flush=True)

        if val_metrics["auc_roc"] > best["auc"]:
            best = {"auc": val_metrics["auc_roc"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= cfg["patience"]:
            print(f"  no val improvement for {cfg['patience']} epochs, stopping")
            break

    train_seconds = time.perf_counter() - train_start
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # ---- evaluation ---- #
    t0 = time.perf_counter()
    test_pred = predict(model, dl_test, device, amp_dtype, tta=cfg["tta"])
    predict_seconds = time.perf_counter() - t0
    val_pred = predict(model, dl_val, device, amp_dtype, tta=cfg["tta"])

    test_pred.to_csv(out_dir / "predictions_test.csv", index=False)
    val_pred.to_csv(out_dir / "predictions_val.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    metrics = {
        "run_id": rid,
        "config": cfg,
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "images_per_second": len(ds_test) / max(1e-9, predict_seconds),
        "device": describe_device(),
        **models.count_parameters(model),
        "test": classification_metrics(test_pred["label"], test_pred["prob"]),
        "val": classification_metrics(val_pred["label"], val_pred["prob"]),
    }
    if cfg["tta"]:
        metrics["test_tta"] = classification_metrics(test_pred["label"], test_pred["prob_tta"])

    # how far from D4-invariant the trained network ended up
    sample = torch.stack([ds_test[i]["image"] for i in range(min(64, len(ds_test)))]).to(device)
    metrics["d4_invariance_error"] = models.d4_invariance_error(model, sample)

    write_json(metrics_path, metrics)
    if cfg["save_checkpoint"]:
        torch.save({"config": cfg, "state_dict": model.state_dict()},
                   out_dir / "checkpoint.pt")

    print(f"  test acc {metrics['test']['accuracy']:.4f}  "
          f"auc {metrics['test']['auc_roc']:.4f}  ece {metrics['test']['ece']:.4f}  "
          f"({train_seconds / 60:.1f} min)")
    return metrics


# --------------------------------------------------------------------------- #
# Cross-survey evaluation
# --------------------------------------------------------------------------- #

def evaluate_decals(rid: str, workers: int = 8) -> dict:
    """Score a trained run on Galaxy10 DECaLS without any adaptation."""
    from src.common import read_json

    out_dir = config.RUNS / rid
    ckpt_path = out_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"{rid} has no checkpoint; retrain it with --save-checkpoint")

    cfg = read_json(out_dir / "metrics.json")["config"]
    device = get_device()
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, norm, _ = models.build(cfg["arch"], input_size=cfg["size"], pretrained=False,
                                  finetune="full",
                                  orientation_pooled=cfg["orientation_pooled"])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    model = model.to(device)

    table = load_decals_table()
    table = table.assign(p_featured=table["label"].astype(np.float32),
                         agreement=1.0)
    ds = GalaxyDataset(table, config.ARRAYS / "decals_images.npy", cfg["size"], "none",
                       norm, train=False)
    dl = make_loader(ds, 256, False, workers)

    pred = predict(model, dl, device, amp_dtype, tta=True)
    pred.to_csv(out_dir / "predictions_decals.csv", index=False)
    out = {
        "decals": classification_metrics(pred["label"], pred["prob"]),
        "decals_tta": classification_metrics(pred["label"], pred["prob_tta"]),
    }
    metrics = read_json(out_dir / "metrics.json")
    metrics.update(out)
    write_json(out_dir / "metrics.json", metrics)
    print(f"{rid}: DECaLS acc {out['decals']['accuracy']:.4f} "
          f"(auc {out['decals']['auc_roc']:.4f})")
    return out


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-csv", type=Path, default=None)
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--arch", choices=models.ARCHITECTURES, default=None)
    ap.add_argument("--label-mode", choices=LABEL_MODES, default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--finetune", choices=models.FINETUNE_MODES, default=None)
    ap.add_argument("--loss", choices=LOSSES, default=None)
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--orientation-pooled", action="store_true")
    ap.add_argument("--random-init", action="store_true",
                    help="skip the ImageNet weights and train the backbone from scratch")
    ap.add_argument("--save-checkpoint", action="store_true")
    ap.add_argument("--workers", type=int, default=int(__import__("os").environ.get("GZM_WORKERS", 8)))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--decals", metavar="RUN_ID", default=None,
                    help="skip training and score an existing run on Galaxy10 DECaLS")
    args = ap.parse_args()

    if args.decals:
        evaluate_decals(args.decals, args.workers)
        return

    if args.jobs_csv is not None:
        if args.task_id is None:
            raise SystemExit("--jobs-csv needs --task-id")
        jobs = pd.read_csv(args.jobs_csv)
        if not 0 <= args.task_id < len(jobs):
            raise SystemExit(f"task id {args.task_id} outside 0..{len(jobs) - 1}")
        row = jobs.iloc[args.task_id].to_dict()
        cfg = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for key in ("size", "train_size", "seed", "epochs", "batch_size"):
            if cfg.get(key) is not None:
                cfg[key] = int(cfg[key])
        cfg["orientation_pooled"] = bool(cfg.get("orientation_pooled", False))
        cfg["save_checkpoint"] = bool(cfg.get("save_checkpoint", False))
        cfg["pretrained"] = bool(cfg.get("pretrained", True))
        print(f"job {args.task_id} of {len(jobs)}: {cfg}")
    else:
        cfg = {
            "arch": args.arch, "label_mode": args.label_mode, "policy": args.policy,
            "size": args.size, "finetune": args.finetune, "loss": args.loss,
            "train_size": args.train_size, "seed": args.seed, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr,
            "orientation_pooled": args.orientation_pooled,
            "save_checkpoint": args.save_checkpoint,
        }
        cfg = {k: v for k, v in cfg.items() if v is not None and v is not False}
        if args.random_init:
            cfg["pretrained"] = False

    train_one(cfg, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
