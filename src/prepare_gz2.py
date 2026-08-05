"""Build the Galaxy Zoo 2 working set: label table, splits and an image cache.

    python -m src.prepare_gz2 --stage table
    python -m src.prepare_gz2 --stage images      # the slow one, run on the cpu queue
    python -m src.prepare_gz2 --stage all

What comes out of it, in $GZM_WORK/arrays:

    gz2_table.csv    one row per galaxy: asset_id, p_featured, agreement, votes,
                     hard label, split, row index into the image array
    gz2_images.npy   uint8 memmap, (N, 160, 160, 3), rows aligned with the table

Three decisions worth spelling out, because the rest of the paper depends on them.

1. The question. We do not use the first letter of `gz2_class`. That string is the
   single class Hart et al. assign to each galaxy and its early-type bin quietly
   absorbs lenticulars and edge-on disks, which is not the question we want to
   ask. Instead we take task 01 of the decision tree ("smooth and rounded, or
   features/disk?") and renormalise its two galaxy answers.

2. Raw fractions, not debiased ones. Hart et al. publish both, and the debiased
   values are the better estimate of intrinsic morphology, so the obvious choice
   looks wrong until you notice two things. First, the debiasing corrects for
   redshift-dependent classification bias, which means the debiased value depends
   on the galaxy's redshift -- information that is simply not in a cutout. Asking
   a network to predict it is asking it to predict something partly unavailable in
   its input. Second, and this is what the paper's argument rests on, the vote
   model needs the label to be a threshold on a *sampled proportion* of a known
   number of draws. A debiased value is no longer a proportion. So `p_featured` is
   the raw fraction; `p_featured_debiased` is kept alongside it for the
   sensitivity check, and it is worth knowing that on this catalogue the two
   correlate at only about 0.74, so the choice is not cosmetic.

3. The crop. GZ2 jpegs are 424x424 with a per-galaxy pixel scale; the target sits
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


def _vote_count_columns(header: list[str]) -> tuple[list[str], str]:
    """Find how many people answered task 01, per galaxy.

    Different releases of the table name the per-task total differently, so rather
    than guessing we prefer the definition: the number of votes on task 01 is the
    sum of the counts of its three answers. Those per-answer `_count` columns are
    the one thing the schema has always had. A published total column is used when
    present, since it is cheaper to read, and the weights are the last resort --
    they are effective sample sizes rather than integers, but they are the right
    order of magnitude for the binomial.
    """
    for name in (f"{config.T01}_total_count", f"{config.T01}_count",
                 f"{config.T01}_total_classifications", "total_classifications",
                 "total_votes"):
        if name in header:
            return [name], f"'{name}'"

    answer_counts = [f"{a}_count" for a in
                     (config.T01_SMOOTH, config.T01_FEATURED, config.T01_ARTIFACT)]
    if all(c in header for c in answer_counts):
        return answer_counts, "the sum of the three task-01 answer counts"

    answer_weights = [f"{a}_weight" for a in
                      (config.T01_SMOOTH, config.T01_FEATURED, config.T01_ARTIFACT)]
    if all(c in header for c in answer_weights):
        return answer_weights, "the sum of the three task-01 answer weights"

    raise SystemExit(
        f"no vote-count column for {config.T01}. Columns starting with "
        f"'{config.T01}': " + ", ".join(c for c in header if c.startswith(config.T01))
    )


def build_table() -> pd.DataFrame:
    if not config.GZ2_CATALOG.exists():
        raise SystemExit(f"missing {config.GZ2_CATALOG}; run src.download_data first")

    header = pd.read_csv(config.GZ2_CATALOG, nrows=0).columns.tolist()
    # The raw fractions are the target: a proportion of a known number of draws,
    # and a function of what the volunteers actually saw. See the module docstring.
    raw_cols, raw_suffix = _fraction_columns(header, ("fraction", "weighted_fraction",
                                                      "debiased"))
    print(f"using '{raw_suffix}' vote fractions as the target and for the vote model")

    # The debiased values are carried along for the sensitivity check only.
    frac_cols, suffix = _fraction_columns(header)
    print(f"carrying '{suffix}' vote fractions for the sensitivity check")

    count_cols, count_how = _vote_count_columns(header)
    print(f"using {count_how} as the vote count")

    usecols = sorted({"dr7objid", "ra", "dec", "gz2_class", *count_cols,
                      *frac_cols.values(), *raw_cols.values()})
    with timer("read catalogue"):
        cat = pd.read_csv(config.GZ2_CATALOG, usecols=usecols)
    print(f"catalogue: {len(cat):,} rows")

    mapping = pd.read_csv(config.GZ2_MAPPING, usecols=["objid", "asset_id"])
    df = cat.merge(mapping, left_on="dr7objid", right_on="objid", how="inner")
    df = df.drop(columns=["objid"])
    print(f"after joining the filename mapping: {len(df):,} rows")

    r_smooth = df[raw_cols["smooth"]].to_numpy(dtype=np.float64)
    r_feat = df[raw_cols["featured"]].to_numpy(dtype=np.float64)
    r_art = df[raw_cols["artifact"]].to_numpy(dtype=np.float64)
    d_smooth = df[frac_cols["smooth"]].to_numpy(dtype=np.float64)
    d_feat = df[frac_cols["featured"]].to_numpy(dtype=np.float64)
    votes = df[count_cols].sum(axis=1).to_numpy(dtype=np.float64)

    galaxy_mass = r_smooth + r_feat
    keep = (
        np.isfinite(r_smooth) & np.isfinite(r_feat) & np.isfinite(r_art)
        & (r_art <= config.ARTIFACT_MAX)
        & (galaxy_mass > 1e-6)
        & (votes >= config.MIN_VOTES)
    )
    dropped = {
        "artifact": int(((r_art > config.ARTIFACT_MAX) & np.isfinite(r_art)).sum()),
        "few_votes": int((votes < config.MIN_VOTES).sum()),
        "no_fraction": int((~np.isfinite(r_smooth) | ~np.isfinite(r_feat)).sum()),
    }
    print(f"dropped {dict(dropped)}")

    df = df.loc[keep].copy()
    p = np.clip(r_feat[keep] / galaxy_mass[keep], 0.0, 1.0)
    debiased_mass = np.clip(d_smooth[keep] + d_feat[keep], 1e-6, None)
    p_debiased = np.clip(d_feat[keep] / debiased_mass, 0.0, 1.0)

    out = pd.DataFrame({
        "asset_id": df["asset_id"].to_numpy(dtype=np.int64),
        "dr7objid": df["dr7objid"].to_numpy(dtype=np.int64),
        "ra": df["ra"].to_numpy(dtype=np.float64),
        "dec": df["dec"].to_numpy(dtype=np.float64),
        "gz2_class": df["gz2_class"].to_numpy(),
        # the target and the quantity everything else is defined from
        "p_featured": p,
        # kept only for the sensitivity check on the vote model
        "p_featured_debiased": p_debiased,
        "votes": votes[keep].astype(np.int32),
        # |2p-1|: 0 when the volunteers split evenly, 1 when they were unanimous
        "agreement": np.abs(2.0 * p - 1.0),
        "label": (p > 0.5).astype(np.int8),
    })
    out = out.sort_values("asset_id").reset_index(drop=True)
    print(f"kept {len(out):,} galaxies "
          f"({100 * out['label'].mean():.1f}% featured); "
          f"raw and debiased thresholds disagree on "
          f"{100 * np.mean((p > 0.5) != (p_debiased > 0.5)):.1f}% of them")
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
    ap.add_argument("--inspect", action="store_true",
                    help="print the task-01 columns the catalogue actually has, and "
                         "which ones we would use, then stop")
    args = ap.parse_args()

    if args.inspect:
        header = pd.read_csv(config.GZ2_CATALOG, nrows=0).columns.tolist()
        print(f"{len(header)} columns in {config.GZ2_CATALOG.name}\n")
        print(f"columns matching '{config.T01}':")
        for c in header:
            if c.startswith(config.T01):
                print(f"  {c}")
        print("\ncolumns that look like a global count:")
        for c in header:
            if "total" in c.lower() or c.lower().endswith("_count"):
                print(f"  {c}")
        print()
        frac_cols, suffix = _fraction_columns(header)
        print(f"target fractions:    {suffix} -> {list(frac_cols.values())}")
        raw_cols, raw_suffix = _fraction_columns(header, ("fraction", "weighted_fraction",
                                                         "debiased"))
        print(f"vote-model fractions: {raw_suffix} -> {list(raw_cols.values())}")
        count_cols, how = _vote_count_columns(header)
        print(f"vote count:          {how} -> {count_cols}")
        return

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
