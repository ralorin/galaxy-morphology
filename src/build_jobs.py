"""Expand the experiment design into one row per training run.

    python -m src.build_jobs                     # everything
    python -m src.build_jobs --groups main labels
    python -m src.build_jobs --list              # just show the group sizes

It writes $GZM_WORK/jobs/jobs.csv and prints the `--array` range to hand to
sbatch. Rows are de-duplicated on the run id, so the groups can overlap (they do:
the label ablation reuses the hard and soft runs from the main comparison) without
training anything twice.

The groups
----------
main         every architecture under one fixed protocol, with hard and with soft
             labels, five seeds. This is the table the paper leads with and the
             sample the Friedman test runs on.
labels       the six label modes on three representative architectures,
             including the two built from debiased rather than raw fractions.
augment      the four augmentation policies. Together with `orientation` this is
             the measurement of how much of the accuracy comes from telling the
             network that orientation does not matter.
orientation  the same architectures wrapped so that they are D4-invariant by
             construction, with and without augmentation on top.
resolution   input size sweep.
finetune     frozen head vs last stage vs everything.
curve        training-set size sweep with hard and with soft labels. The gap
             between the two curves, and where each one flattens, is the evidence
             for the label-noise account of the accuracy ceiling.
loss         plain BCE vs focal vs label smoothing.
pretraining  ImageNet weights vs random initialisation, both fine-tuned end to
             end. This is the comparison our earlier study got wrong, and it is
             redone here without the frozen-random-backbone confound.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

import config
from src.common import ensure_dir
from src.experiment import default_config, resolve, run_id
from src.registry import ARCHITECTURES, FINETUNE_MODES, REGISTRY

# The protocol everything else is measured against.
REFERENCE = dict(policy="d4", finetune="full", loss="bce", size=None, train_size=None)

# Three architectures carry the ablations: one classic residual net, one modern
# convnet and one transformer. Running the full grid on all thirteen would cost
# four times as much for the same conclusions.
PROBES = ("resnet50", "convnext_tiny", "vit_small")

CURVE_SIZES = (2000, 5000, 10000, 25000, 50000, None)
CURVE_PROBES = ("resnet50", "vit_small")


def _job(**kw) -> dict:
    job = {**{k: v for k, v in default_config().items()
              if k in ("arch", "label_mode", "policy", "size", "finetune", "loss",
                       "train_size", "seed", "orientation_pooled", "pretrained",
                       "epochs", "batch_size", "save_checkpoint")}}
    job.update(REFERENCE)
    job.update(kw)
    return job


def group_main() -> list[dict]:
    jobs = []
    for arch, mode, seed in itertools.product(ARCHITECTURES, ("hard", "soft"),
                                              config.SEEDS):
        jobs.append(_job(arch=arch, label_mode=mode, seed=seed,
                         # seed 0 keeps its weights: the explanation analysis and
                         # the cross-survey evaluation both need them
                         save_checkpoint=(seed == 0)))
    return jobs


def group_labels() -> list[dict]:
    from src.experiment import LABEL_MODES

    return [_job(arch=a, label_mode=m, seed=s)
            for a, m, s in itertools.product(PROBES, LABEL_MODES, (0, 1, 2))]


def group_augment() -> list[dict]:
    from src.experiment import POLICIES

    return [_job(arch=a, policy=p, label_mode="soft", seed=s)
            for a, p, s in itertools.product(PROBES, POLICIES, (0, 1, 2))]


def group_orientation() -> list[dict]:
    """Invariance built into the network, against invariance learnt from data.

    These are the most expensive runs in the design: pooling over the eight group
    elements costs eight forward and backward passes per step. We therefore keep
    the group as small as the question allows.

    In particular there is no `d4` augmentation variant here. Augmenting over a
    group the network is already exactly invariant to cannot add information; it
    can only change the order in which samples arrive. The claim is checkable for
    free rather than by spending GPU time on it, because every run reports
    `d4_invariance_error`, which is zero by construction for these models.

    The cheap form of the same idea, one ordinary network with eight views averaged
    at inference, is not in this group either, because it is computed for every
    run in the study as the TTA column.
    """
    return [_job(arch=a, policy="none", label_mode="soft", seed=s,
                 orientation_pooled=True, save_checkpoint=(s == 0))
            for a, s in itertools.product(PROBES, (0, 1, 2))]


def group_resolution() -> list[dict]:
    jobs = []
    for arch, seed in itertools.product(PROBES, (0, 1)):
        for size in (112, 160, 224):
            if REGISTRY[arch].fixed_size and size != REGISTRY[arch].input_size:
                continue
            jobs.append(_job(arch=arch, size=size, label_mode="soft", seed=seed))
    # the scratch nets live at a smaller scale, sweep them separately
    for seed in (0, 1):
        for size in (64, 96, 128):
            jobs.append(_job(arch="cnn_small", size=size, label_mode="soft", seed=seed))
    return jobs


def group_finetune() -> list[dict]:
    return [_job(arch=a, finetune=f, label_mode="soft", seed=s)
            for a, f, s in itertools.product(PROBES, FINETUNE_MODES, (0, 1))]


def group_curve() -> list[dict]:
    return [_job(arch=a, train_size=n, label_mode=m, seed=s)
            for a, n, m, s in itertools.product(CURVE_PROBES, CURVE_SIZES,
                                                ("hard", "soft"), (0, 1))]


def group_loss() -> list[dict]:
    from src.experiment import LOSSES

    return [_job(arch=a, loss=l, label_mode="soft", seed=s)
            for a, l, s in itertools.product(PROBES, LOSSES, (0, 1))]


def group_pretraining() -> list[dict]:
    """ImageNet initialisation against random initialisation, both fine-tuned end
    to end.

    Our earlier comparison concluded that ImageNet pretraining does not help on
    galaxy images, but it reached that conclusion with the backbone frozen and its
    weights random, which trains a head on random features and is not a test of
    pretraining at all. This group repeats the comparison properly, and the
    `finetune` group above separates the frozen case out on its own.
    """
    return [_job(arch=a, pretrained=p, label_mode="soft", seed=s, finetune="full")
            for a, p, s in itertools.product(PROBES, (True, False), (0, 1, 2))]


GROUPS = {
    "main": group_main,
    "labels": group_labels,
    "augment": group_augment,
    "orientation": group_orientation,
    "resolution": group_resolution,
    "finetune": group_finetune,
    "curve": group_curve,
    "loss": group_loss,
    "pretraining": group_pretraining,
}


def build(groups: list[str]) -> pd.DataFrame:
    rows, seen = [], {}
    for name in groups:
        for job in GROUPS[name]():
            rid = run_id(resolve(dict(job)))
            if rid in seen:
                # keep the checkpoint flag if any group asked for it
                if job.get("save_checkpoint"):
                    rows[seen[rid]]["save_checkpoint"] = True
                rows[seen[rid]]["group"] = rows[seen[rid]]["group"] + "+" + name
                continue
            seen[rid] = len(rows)
            rows.append({"run_id": rid, "group": name, **job})
    df = pd.DataFrame(rows)
    # Put the long runs first. SLURM starts array tasks in order, so the tail of
    # the array is short jobs that slot into whatever time is left.
    order = {"transformer": 0, "cnn": 1, "custom": 2}
    df["_family"] = df["arch"].map(lambda a: order[REGISTRY[a].family])
    df = df.sort_values(["_family", "arch", "seed"]).drop(columns="_family")
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="*", choices=sorted(GROUPS), default=sorted(GROUPS))
    ap.add_argument("--out", type=Path, default=config.JOBS_CSV)
    ap.add_argument("--list", action="store_true", help="print group sizes and exit")
    args = ap.parse_args()

    if args.list:
        total = 0
        for name, fn in GROUPS.items():
            n = len({run_id(resolve(dict(j))) for j in fn()})
            total += n
            print(f"  {name:12s} {n:4d} distinct runs")
        print(f"  {'union':12s} {len(build(sorted(GROUPS))):4d} distinct runs "
              f"({total} before de-duplication)")
        return

    df = build(args.groups)
    ensure_dir(args.out.parent)
    df.to_csv(args.out, index=False)

    print(f"wrote {len(df)} jobs to {args.out}")
    print(df.groupby("group").size().to_string())
    print()
    print(f"submit with:  sbatch --array=0-{len(df) - 1}%2 slurm/02_train_array.sh")
    print("(%2 because gpu-small allows two GPUs per user; raise it if the limit changes)")


if __name__ == "__main__":
    main()
