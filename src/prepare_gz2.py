"""Build the Galaxy Zoo 2 working set: label table, splits and an image cache.

    python -m src.prepare_gz2 --stage table
    python -m src.prepare_gz2 --stage images      # the slow one, run on the cpu queue
    python -m src.prepare_gz2 --stage all

What comes out of it, in $GZM_WORK/arrays:

    gz2_table.csv    one row per galaxy: asset_id, p_featured, agreement, votes,
                     hard label, split, row index into the image array
    gz2_images.npy   uint8 memmap, (N, 160, 160, 3), rows aligned with the table

Two decisions worth spelling out, because the rest of the paper depends on them.

1. The label. We do not use the first letter of `gz2_class`. That string is the
   single class Hart et al. assign to each galaxy and its early-type bin quietly
   absorbs lenticulars and edge-on disks, which is not the question we want to
   ask. Instead we take task 01 of the decision tree ("smooth and rounded, or
   features/disk?"), renormalise its two galaxy answers, and keep the resulting
   fraction as a continuous target. The volunteers answered exactly this
   question, so the fraction is a direct measure of how much they agreed.

2. The crop. GZ2 jpegs are 424x424 with a per-galaxy pixel scale; the target sits
   in the centre and the corners are full of unrelated objects. We take the
   central 224x224 and cache at 160x160, following the crop-and-downsample recipe
   of Dieleman et al. (2015). Models resize from the cache to whatever input they
   want, so there is one array on disk instead of one per resolution.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.common import ensure_dir, timer, write_json

TABLE = config.ARRAYS / "gz2_table.csv"
IMAGES = config.ARRAYS / "gz2_images.npy"
META = config.ARRAYS / "gz2_meta.json"


# --------------------------------------------------------------------------- #
# Label table
# --------------------------------------------------------------------------- #

def _fraction_columns(header: list[str], suffixes=("debiased", "weighted_fraction",
                                                   "fraction")) -> tuple[dict, str]:
    """Pick a set of task-01 fraction columns, preferring the first suffix given.

    Hart et al. (2016) publish `*_debiased` fractions; older copies of the table
    only carry `*_weighted_fraction`. Both are usable, they are just not the same
    numbers, so we record which one we read.
    """
    for suffix in suffixes:
        cols = {
            "smooth": f"{config.T01_SMOOTH}_{suffix}",
            "featured": f"{config.T01_FEATURED}_{suffix}",
            "artifact": f"{config.T01_ARTIFACT}_{suffix}",
        }
        if all(c in header for c in cols.values()):
            return cols, suffix
    raise SystemExit(
        "could not find task-01 vote fractions in the catalogue. Columns starting "
        f"with '{config.T01}' present: "
        + ", ".join(c for c in header if c.startswith(config.T01))[:400]
    )


def build_table() -> pd.DataFrame:
    if not config.GZ2_CATALOG.exists():
        raise SystemExit(f"missing {config.GZ2_CATALOG}; run src.download_data first")

    header = pd.read_csv(config.GZ2_CATALOG, nrows=0).columns.tolist()
    frac_cols, suffix = _fraction_columns(header)
    print(f"using '{suffix}' vote fractions for task 01 (the training target)")

    # The raw, unweighted fractions are what the binomial vote model in the
    # analysis needs: they are counts over a known number of votes, whereas the
    # debiased values have been corrected for redshift-dependent bias and are no
    # longer a simple proportion.
    raw_cols, raw_suffix = _fraction_columns(header, ("fraction", "weighted_fraction",
                                                      "debiased"))
    print(f"using '{raw_suffix}' vote fractions for the vote model")

    count_col = next((c for c in (f"{config.T01}_total_count",
                                  f"{config.T01}_count",
                                  f"{config.T01}_total_weight") if c in header), None)
    if count_col is None:
        raise SystemExit(f"no vote-count column for {config.T01}")
    print(f"using '{count_col}' as the vote count")

    usecols = sorted({"dr7objid", "ra", "dec", "gz2_class", count_col,
                      *frac_cols.values(), *raw_cols.values()})
    with timer("read catalogue"):
        cat = pd.read_csv(config.GZ2_CATALOG, usecols=usecols)
    print(f"catalogue: {len(cat):,} rows")

    mapping = pd.read_csv(config.GZ2_MAPPING, usecols=["objid", "asset_id"])
    df = cat.merge(mapping, left_on="dr7objid", right_on="objid", how="inner")
    df = df.drop(columns=["objid"])
    print(f"after joining the filename mapping: {len(df):,} rows")

    f_smooth = df[frac_cols["smooth"]].to_numpy(dtype=np.float64)
    f_feat = df[frac_cols["featured"]].to_numpy(dtype=np.float64)
    f_art = df[frac_cols["artifact"]].to_numpy(dtype=np.float64)
    r_smooth = df[raw_cols["smooth"]].to_numpy(dtype=np.float64)
    r_feat = df[raw_cols["featured"]].to_numpy(dtype=np.float64)
    votes = df[count_col].to_numpy(dtype=np.float64)

    galaxy_mass = f_smooth + f_feat
    keep = (
        np.isfinite(f_smooth) & np.isfinite(f_feat) & np.isfinite(f_art)
        & (f_art <= config.ARTIFACT_MAX)
        & (galaxy_mass > 1e-6)
        & (votes >= config.MIN_VOTES)
    )
    dropped = {
        "artifact": int(((f_art > config.ARTIFACT_MAX) & np.isfinite(f_art)).sum()),
        "few_votes": int((votes < config.MIN_VOTES).sum()),
        "no_fraction": int((~np.isfinite(f_smooth) | ~np.isfinite(f_feat)).sum()),
    }
    print(f"dropped {dict(dropped)}")

    df = df.loc[keep].copy()
    p = f_feat[keep] / galaxy_mass[keep]
    raw_mass = np.clip(r_smooth[keep] + r_feat[keep], 1e-6, None)
    p_raw = np.clip(r_feat[keep] / raw_mass, 0.0, 1.0)

    out = pd.DataFrame({
        "asset_id": df["asset_id"].to_numpy(dtype=np.int64),
        "dr7objid": df["dr7objid"].to_numpy(dtype=np.int64),
        "ra": df["ra"].to_numpy(dtype=np.float64),
        "dec": df["dec"].to_numpy(dtype=np.float64),
        "gz2_class": df["gz2_class"].to_numpy(),
        "p_featured": p,
        "p_featured_raw": p_raw,
        "votes": votes[keep].astype(np.int32),
        # |2p-1|: 0 when the volunteers split evenly, 1 when they were unanimous
        "agreement": np.abs(2.0 * p - 1.0),
        "label": (p > 0.5).astype(np.int8),
    })
    out = out.sort_values("asset_id").reset_index(drop=True)
    print(f"kept {len(out):,} galaxies "
          f"({100 * out['label'].mean():.1f}% featured)")
    return out


def add_splits(df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Stratified 70/10/20 split.

    We stratify on the label crossed with the agreement bin, not just on the
    label. Otherwise a random split leaves a different amount of ambiguous
    galaxies in each part and the agreement-conditioned analysis wobbles between
    seeds for no good reason.
    """
    rng = np.random.default_rng(seed)
    bins = np.digitize(df["agreement"].to_numpy(), np.array(config.AGREEMENT_BINS[1:-1]))
    stratum = df["label"].to_numpy().astype(int) * 10 + bins

    split = np.empty(len(df), dtype=object)
    f_train, f_val, _ = config.SPLIT_FRACTIONS
    for s in np.unique(stratum):
        idx = np.nonzero(stratum == s)[0]
        rng.shuffle(idx)
        n_train = int(round(f_train * idx.size))
        n_val = int(round(f_val * idx.size))
        split[idx[:n_train]] = "train"
        split[idx[n_train:n_train + n_val]] = "val"
        split[idx[n_train + n_val:]] = "test"

    df = df.copy()
    df["split"] = split
    df["row"] = np.arange(len(df), dtype=np.int64)
    print(df.groupby("split")["label"].agg(["size", "mean"]))
    return df


# --------------------------------------------------------------------------- #
# Image cache
# --------------------------------------------------------------------------- #

def find_image_root() -> Path:
    """The Zenodo archive nests the jpegs one or two levels deep; locate them."""
    if not config.GZ2_IMAGES.exists():
        raise SystemExit(f"missing {config.GZ2_IMAGES}; run src.download_data first")
    for candidate in (config.GZ2_IMAGES / "images", config.GZ2_IMAGES):
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate
    for path in config.GZ2_IMAGES.rglob("*.jpg"):
        return path.parent
    raise SystemExit(f"no jpegs found under {config.GZ2_IMAGES}")


def _load_one(args):
    """Read one jpeg, centre-crop, resize. Returns (row, array) or (row, None)."""
    import cv2

    row, path, crop, size = args
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return row, None
    h, w = img.shape[:2]
    top = (h - crop) // 2
    left = (w - crop) // 2
    if top < 0 or left < 0:  # smaller than the crop: resize the whole frame
        patch = img
    else:
        patch = img[top:top + crop, left:left + crop]
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return row, cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)


def build_images(df: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    root = find_image_root()
    print(f"reading jpegs from {root}")
    size = config.CACHE_SIZE
    ensure_dir(config.ARRAYS)

    arr = np.lib.format.open_memmap(
        IMAGES, mode="w+", dtype=np.uint8, shape=(len(df), size, size, 3))

    jobs = [(int(r.row), root / f"{int(r.asset_id)}.jpg", config.GZ2_CROP, size)
            for r in df.itertuples()]
    workers = workers or int(os.environ.get("GZM_WORKERS", os.cpu_count() or 4))
    print(f"decoding {len(jobs):,} images with {workers} workers")

    missing = []
    with timer("decode"), ProcessPoolExecutor(max_workers=workers) as pool:
        for n, (row, patch) in enumerate(pool.map(_load_one, jobs, chunksize=256), 1):
            if patch is None:
                missing.append(row)
            else:
                arr[row] = patch
            if n % 20000 == 0:
                print(f"  {n:,} / {len(jobs):,}", flush=True)
    arr.flush()

    if missing:
        # A few hundred rows of the catalogue have no image in the archive (the
        # Zenodo README mentions a ~0.08% mismatch). Drop them from the table
        # rather than leaving zero-filled rows in the array.
        print(f"{len(missing)} catalogue rows had no image; dropping them")
        df = df[~df["row"].isin(set(missing))].reset_index(drop=True)

    return df


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["table", "images", "all"], default="all")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    ensure_dir(config.ARRAYS)

    if args.stage in ("table", "all"):
        df = add_splits(build_table())
        df.to_csv(TABLE, index=False)
        print(f"wrote {TABLE}")
    else:
        df = pd.read_csv(TABLE)

    if args.stage in ("images", "all"):
        df = build_images(df, args.workers)
        df.to_csv(TABLE, index=False)

    write_json(META, {
        "n_galaxies": int(len(df)),
        "cache_size": config.CACHE_SIZE,
        "crop": config.GZ2_CROP,
        "min_votes": config.MIN_VOTES,
        "artifact_max": config.ARTIFACT_MAX,
        "split_counts": df["split"].value_counts().to_dict(),
        "featured_fraction": float(df["label"].mean()),
        "median_votes": float(df["votes"].median()),
        "mean_agreement": float(df["agreement"].mean()),
    })
    print(f"wrote {META}")


if __name__ == "__main__":
    main()
