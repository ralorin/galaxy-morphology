"""Build a Fashion-MNIST-H working set: one binary pair, with the votes kept.

    python -m src.prepare_fmh --inspect
    python -m src.prepare_fmh --pair pullover coat

Outputs in $GZM_WORK/arrays:

    fmh_table.csv    one row per image: p, agreement, votes, label, split, row index
    fmh_images.npy   uint8 array, (N, 28, 28, 3), rows aligned with the table
    fmh_meta.json    the same kind of record prepare_gz2 writes for Galaxy Zoo 2

Why this dataset. The argument is about labels that are panel verdicts, and it is
made on Galaxy Zoo 2. Fashion-MNIST-H (Ishida et al., ICLR 2023) collects around
sixty-seven annotations for each of the ten thousand Fashion-MNIST test images, which
is the closest thing outside astronomy to the vote record we rely on. Its confusable
classes disagree far more than CIFAR-10H does: over the ten classes the panel ceiling
is about 96%, against 99.7% for CIFAR-10H, and for pullover against coat it falls to
about 92%.

The votes cover the test split only, so training uses the ordinary Fashion-MNIST
labels and only the evaluation is agreement-resolved. That is the half of the
argument this replication is for: where the residual error lands, and how much of it
the annotation accounts for.

The two-answer renormalisation follows the label definition in the paper. An image
counts towards the pair when its gold label is one of the two and when the two
answers between them hold at least --min-share of the panel, which plays the part
that discarding the artefact votes plays on Galaxy Zoo 2.
"""

from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.common import ensure_dir

CLASSES = ["t-shirt", "trouser", "pullover", "dress", "coat",
           "sandal", "shirt", "sneaker", "bag", "ankle boot"]

FASHION = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
COUNTS = "https://raw.githubusercontent.com/ishida-lab/irreducible/main/data/fmh_counts.csv"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def fetch(url: str, path: Path) -> Path:
    if path.exists():
        return path
    ensure_dir(path.parent)
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, path)
    return path


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic = int.from_bytes(fh.read(4), "big")
        shape = [int.from_bytes(fh.read(4), "big") for _ in range(magic & 0xFF)]
        return np.frombuffer(fh.read(), dtype=np.uint8).reshape(shape)


def load_source() -> dict:
    raw = config.DATA / "fashion_mnist"
    out = {k: read_idx(fetch(f"{FASHION}/{v}", raw / v)) for k, v in FILES.items()}
    out["counts"] = np.loadtxt(fetch(COUNTS, raw / "fmh_counts.csv"),
                               delimiter=",").astype(np.int32)
    return out


def inspect(src: dict) -> None:
    counts, gold = src["counts"], src["test_labels"]
    n = counts.sum(1)
    print(f"test images {len(gold):,}, annotations each: median {int(np.median(n))}, "
          f"min {n.min()}, max {n.max()}")
    print("the panel's top answer takes a median of "
          f"{np.median(counts.max(1) / n):.3f} of the vote\n")
    scored = []
    for a in range(10):
        for b in range(a + 1, 10):
            sel = np.isin(gold, (a, b))
            two = counts[sel, a] + counts[sel, b]
            keep = two / n[sel] >= 0.7
            if keep.sum() < 150:
                continue
            p = 1.0 - counts[sel, a][keep] / two[keep]
            scored.append((100 * float((np.abs(2 * p - 1) < 0.6).mean()),
                           int(keep.sum()), CLASSES[a], CLASSES[b]))
    print(f"{'pair':26s} {'items':>6s} {'contested':>10s}")
    for share, items, a, b in sorted(scored, reverse=True)[:8]:
        print(f"{a + ' vs ' + b:26s} {items:6d} {share:9.1f}%")


def build(src: dict, first: str, second: str, min_share: float,
          val_fraction: float = 0.1):
    a, b = CLASSES.index(first), CLASSES.index(second)
    counts, gold_test = src["counts"], src["test_labels"]
    n_votes = counts.sum(1)

    sel = np.flatnonzero(np.isin(gold_test, (a, b)))
    va = counts[sel, a].astype(np.float64)
    vb = counts[sel, b].astype(np.float64)
    two = va + vb
    keep = two / n_votes[sel] >= min_share
    sel, vb, two = sel[keep], vb[keep], two[keep]
    p_test = vb / two

    gold_train = src["train_labels"]
    train_sel = np.flatnonzero(np.isin(gold_train, (a, b)))
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(train_sel)
    cut = int(len(train_sel) * (1 - val_fraction))

    images = np.concatenate([src["train_images"][train_sel], src["test_images"][sel]])
    images = np.repeat(images[..., None], 3, axis=-1)      # grey into three channels

    p_train = (gold_train[train_sel] == b).astype(np.float64)
    table = pd.DataFrame({
        "p_featured": np.concatenate([p_train, p_test]),
        "votes": np.concatenate([np.zeros(len(train_sel), dtype=int), n_votes[sel]]),
        "votes_binary": np.concatenate([np.zeros(len(train_sel), dtype=int),
                                        two.astype(int)]),
        "split": ["train"] * cut + ["val"] * (len(train_sel) - cut)
                 + ["test"] * len(sel),
    })
    table["agreement"] = np.abs(2 * table["p_featured"] - 1.0)
    table["label"] = (table["p_featured"] > 0.5).astype(int)
    table["row"] = np.arange(len(table))

    test = table[table["split"] == "test"]
    meta = {
        "dataset": "fashion-mnist-h",
        "pair": [first, second],
        "positive_class": second,
        "min_share": min_share,
        "n_items": int(len(table)),
        "split_counts": {s: int((table["split"] == s).sum())
                         for s in ("train", "val", "test")},
        "featured_fraction": float(test["label"].mean()),
        "majority_baseline": float(max(test["label"].mean(),
                                       1 - test["label"].mean())),
        "median_votes": float(test["votes"].median()),
        "median_votes_binary": float(test["votes_binary"].median()),
        "mean_agreement": float(test["agreement"].mean()),
        "source": {"images": FASHION, "annotations": COUNTS},
    }
    return table, images, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, default=["pullover", "coat"],
                    metavar=("CLASS", "CLASS"))
    ap.add_argument("--min-share", type=float, default=0.7,
                    help="the two answers must hold at least this much of the panel")
    ap.add_argument("--inspect", action="store_true",
                    help="report the disagreement of every pair and stop")
    args = ap.parse_args()

    src = load_source()
    if args.inspect:
        inspect(src)
        return

    for name in args.pair:
        if name not in CLASSES:
            raise SystemExit(f"unknown class {name!r}; known: {', '.join(CLASSES)}")

    table, images, meta = build(src, args.pair[0], args.pair[1], args.min_share)
    out = ensure_dir(config.ARRAYS)
    table.to_csv(out / "fmh_table.csv", index=False)
    np.save(out / "fmh_images.npy", images)
    with open(out / "fmh_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    counts = meta["split_counts"]
    print(f"{meta['pair'][0]} against {meta['pair'][1]}")
    print(f"  train {counts['train']:,} | val {counts['val']:,} | "
          f"test {counts['test']:,}, and only the test split carries votes")
    print(f"  median panel {meta['median_votes_binary']:.0f} on the two answers, "
          f"{meta['median_votes']:.0f} overall")
    print(f"  {meta['positive_class']} is {100 * meta['featured_fraction']:.1f}% of "
          f"the test split, so a constant classifier scores "
          f"{100 * meta['majority_baseline']:.1f}%")
    print(f"  wrote {out / 'fmh_table.csv'} and {out / 'fmh_images.npy'}")


if __name__ == "__main__":
    main()
