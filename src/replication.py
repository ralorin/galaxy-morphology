"""The second dataset: does the label ceiling argument hold away from astronomy?

    python -m src.replication

Reads the Fashion-MNIST-H runs from $GZM_WORK/runs and writes

    results/replication.json      the ceiling, the agreement profile, the error budget
    results/replication.csv       one row per run and agreement bin

Two datasets are checked and they answer differently, which is the point.

Fashion-MNIST-H carries about sixty-seven annotations for each test image and its
confusable classes genuinely divide the panel, so it has a ceiling below 100% and a
contested tail, like Galaxy Zoo 2. CIFAR-10H carries about fifty and almost never
divides: its ten-class ceiling is 99.7%, and under one item in a hundred is
contested. Reporting both is worth more than reporting either. The first says the
argument travels; the second says the instrument is not vacuous, since on a dataset
whose annotation is nearly unanimous it correctly reports that there is nothing to
find.
"""

from __future__ import annotations

import argparse
import urllib.request

import numpy as np
import pandas as pd

import config
from src.analysis import vote_ceiling
from src.common import ensure_dir, read_json, write_json

CIFAR10H = ("https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/"
            "data/cifar10h-counts.npy")


def multiclass_ceiling(counts: np.ndarray, draws: int = 400, seed: int = 0) -> dict:
    """The same ceiling for a panel with more than two answers.

    The binary case has a closed form; this one does not, so a fresh panel is drawn
    from the observed vote fractions and we count how often its argmax matches the
    recorded label. The paper says the model generalises this way and this is what
    that sentence means in code.
    """
    n = counts.sum(1)
    p = counts / n[:, None]
    recorded = counts.argmax(1)
    rng = np.random.default_rng(seed)
    agree = np.empty(len(p))
    for i in range(len(p)):
        sample = rng.multinomial(int(n[i]), p[i], size=draws)
        agree[i] = (sample.argmax(1) == recorded[i]).mean()
    return {
        "n_items": int(len(p)),
        "median_votes": float(np.median(n)),
        "bayes_accuracy": float(agree.mean()),
        "contested_share": float((p.max(1) < 0.8).mean()),
    }


def cifar10h_control() -> dict:
    """CIFAR-10H, as the negative control. Downloads 800 kB and needs no images."""
    path = config.DATA / "cifar10h" / "cifar10h-counts.npy"
    if not path.exists():
        ensure_dir(path.parent)
        print(f"  downloading {CIFAR10H}")
        urllib.request.urlretrieve(CIFAR10H, path)
    out = multiclass_ceiling(np.load(path))
    print(f"  CIFAR-10H: A* = {100 * out['bayes_accuracy']:.2f}% over "
          f"{out['n_items']:,} items, {100 * out['contested_share']:.1f}% contested")
    return out


def profile(runs: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Accuracy inside each agreement bin, and each bin's share of the errors."""
    test = table[table["split"] == "test"].reset_index(drop=True)
    edges = np.asarray(config.AGREEMENT_BINS, dtype=float)
    binned = np.clip(np.digitize(test["agreement"].to_numpy(), edges[1:-1]),
                     0, len(edges) - 2)
    rows = []
    for run_id in runs["run_id"]:
        path = config.RUNS / run_id / "predictions_test.csv"
        if not path.exists():
            print(f"  no predictions for {run_id}")
            continue
        pred = pd.read_csv(path)
        correct = ((pred["prob"].to_numpy() >= 0.5).astype(int)
                   == pred["label"].to_numpy().astype(int))
        errors = (~correct).sum()
        for b in range(len(config.AGREEMENT_BINS) - 1):
            inside = binned == b
            if not inside.any():
                continue
            labels = pred["label"].to_numpy()[inside]
            base = max(labels.mean(), 1 - labels.mean())
            rows.append({
                "run_id": run_id, "bin": b,
                "n": int(inside.sum()),
                "share": float(inside.mean()),
                "accuracy": float(correct[inside].mean()),
                "majority_baseline": float(base),
                "lift": float(correct[inside].mean() - base),
                "error_share": float((~correct[inside]).sum() / errors) if errors else 0.0,
            })
    return pd.DataFrame(rows).merge(
        runs[["run_id", "arch", "label_mode", "seed", "test_accuracy"]],
        on="run_id", how="left")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-control", action="store_true",
                    help="do not download or recompute the CIFAR-10H control")
    args = ap.parse_args()

    table_path = config.ARRAYS / "fmh_table.csv"
    if not table_path.exists():
        raise SystemExit(f"missing {table_path}; run src.prepare_fmh first")
    table = pd.read_csv(table_path)
    meta = read_json(config.ARRAYS / "fmh_meta.json")

    ceiling = vote_ceiling(table, "p_featured")
    print(f"  Fashion-MNIST-H {' vs '.join(meta['pair'])}: "
          f"A* = {100 * ceiling['bayes_accuracy']:.2f}% over "
          f"{ceiling['n_test']:,} test items")

    runs = pd.read_csv(config.RESULTS / "runs.csv")
    runs = runs[runs.get("dataset", "gz2") == "fmh"] if "dataset" in runs else runs.iloc[:0]
    if runs.empty:
        raise SystemExit("no replication runs in runs.csv; run slurm/06_replication.sh "
                         "and then src.analysis")
    print(f"  {len(runs)} replication runs over {runs['arch'].nunique()} architectures")

    per_bin = profile(runs, table)
    ensure_dir(config.RESULTS)
    per_bin.to_csv(config.RESULTS / "replication.csv", index=False)

    lowest = per_bin[per_bin["bin"] == 0]
    highest = per_bin[per_bin["bin"] == per_bin["bin"].max()]
    summary = {
        "dataset": meta,
        "ceiling": ceiling,
        "best_accuracy": float(runs["test_accuracy"].max()),
        "mean_accuracy": float(runs["test_accuracy"].mean()),
        "architecture_spread": float(runs.groupby("arch")["test_accuracy"].mean().max()
                                     - runs.groupby("arch")["test_accuracy"].mean().min()),
        "accuracy_low_agreement": float(lowest["accuracy"].mean()),
        "accuracy_high_agreement": float(highest["accuracy"].mean()),
        "share_low_agreement": float(lowest["share"].mean()),
        "error_share_low_agreement": float(lowest["error_share"].mean()),
        "control_cifar10h": None if args.skip_control else cifar10h_control(),
    }
    write_json(config.RESULTS / "replication.json", summary)

    gap = summary["accuracy_high_agreement"] - summary["accuracy_low_agreement"]
    print()
    print(f"  accuracy {100 * summary['accuracy_high_agreement']:.1f}% where the panel "
          f"agreed against {100 * summary['accuracy_low_agreement']:.1f}% where it did not")
    print(f"  the contested {100 * summary['share_low_agreement']:.0f}% of items carry "
          f"{100 * summary['error_share_low_agreement']:.0f}% of the errors")
    print(f"  spread across agreement bins {100 * gap:.1f} points, across architectures "
          f"{100 * summary['architecture_spread']:.1f}")
    print(f"  wrote {config.RESULTS / 'replication.json'}")


if __name__ == "__main__":
    main()
