"""Paths, constants and the small amount of global configuration the pipeline needs.

Everything is driven by two environment variables so that the same code runs on a
laptop and on the cluster without edits:

    GZM_DATA   where the raw downloads live (catalogues, image archives)
    GZM_WORK   where everything we produce goes (arrays, checkpoints, results)

Defaults point at $HOME/galaxy-morphology/{data,work}. Run `python config.py` to
print the resolved paths and check what is already in place.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

HOME = Path(os.environ.get("HOME", Path.home()))

DATA = Path(os.environ.get("GZM_DATA", HOME / "galaxy-morphology" / "data"))
WORK = Path(os.environ.get("GZM_WORK", HOME / "galaxy-morphology" / "work"))

# raw inputs
GZ2_CATALOG = DATA / "gz2_hart16.csv"                  # Hart et al. (2016) debiased table
GZ2_MAPPING = DATA / "gz2_filename_mapping.csv"        # dr7objid -> asset_id
GZ2_IMAGES = DATA / "images_gz2"                       # unpacked images_gz2.zip
DECALS_H5 = DATA / "Galaxy10_DECals.h5"                # astroNN Galaxy10 DECaLS

# derived data
ARRAYS = WORK / "arrays"                               # uint8 memmaps + label tables
SPLITS = WORK / "splits"
RUNS = WORK / "runs"                                   # one directory per training run
RESULTS = WORK / "results"                             # aggregated tables and json
FIGURES = WORK / "figures"
JOBS_CSV = WORK / "jobs" / "jobs.csv"

# --------------------------------------------------------------------------- #
# Download locations (all public, all free to redistribute by their owners)
# --------------------------------------------------------------------------- #

URLS = {
    # Galaxy Zoo 2 debiased catalogue (Hart et al. 2016), served by the GZ team
    "gz2_hart16": "https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz",
    # Galaxy Zoo 2 images from the original sample, Zenodo 3565489 (CC BY 4.0)
    "gz2_mapping": "https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv",
    "gz2_images": "https://zenodo.org/records/3565489/files/images_gz2.zip",
    # Galaxy10 DECaLS, Zenodo 10845026
    "decals": "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5",
}

# --------------------------------------------------------------------------- #
# Image geometry
# --------------------------------------------------------------------------- #

# GZ2 jpegs are 424x424 with a per-galaxy pixel scale, so the target galaxy always
# sits in the middle and neighbours creep in at the edges. Cropping the central
# 224x224 and downsampling is the convention introduced by Dieleman et al. (2015)
# and it is what we follow here. We keep the cache at 160 px and let each model
# resize from there, which avoids storing one array per input resolution.
GZ2_NATIVE = 424
GZ2_CROP = 224
CACHE_SIZE = 160

# Galaxy10 DECaLS ships 256x256 cutouts at a fixed 0.262"/px. Cropping the central
# 224 keeps roughly the same field of view as the GZ2 crops.
DECALS_NATIVE = 256
DECALS_CROP = 224

# --------------------------------------------------------------------------- #
# Label definition
# --------------------------------------------------------------------------- #

# Galaxy Zoo 2 task 01 ("Is the galaxy simply smooth and rounded, or does it have
# features or a disk?"). We use its three answers directly instead of the E/S
# prefix of gz2_class: the prefix mixes lenticulars and edge-on disks into the
# early-type bin, whereas task 01 is exactly the binary question we want to model
# and is the one the volunteers actually answered.
T01 = "t01_smooth_or_features"
T01_SMOOTH = f"{T01}_a01_smooth"
T01_FEATURED = f"{T01}_a02_features_or_disk"
T01_ARTIFACT = f"{T01}_a03_star_or_artifact"

# Objects the volunteers mostly flagged as stars or imaging artefacts are dropped.
ARTIFACT_MAX = 0.3
# Task 01 is the first question so everybody answers it, but a handful of objects
# were seen by very few people; those vote fractions are too noisy to use.
MIN_VOTES = 20

CLASS_NAMES = ("smooth", "featured")  # 0, 1

# --------------------------------------------------------------------------- #
# Experiment defaults
# --------------------------------------------------------------------------- #

SEED = 42
SEEDS = (0, 1, 2, 3, 4)
SPLIT_FRACTIONS = (0.70, 0.10, 0.20)  # train / val / test

# Agreement bins used throughout the analysis. `agreement` is |2p - 1| where p is
# the debiased featured-vs-smooth vote fraction, so 0 means the volunteers were
# split down the middle and 1 means they were unanimous.
AGREEMENT_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _describe(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_dir():
        n = sum(1 for _ in path.iterdir())
        return f"directory, {n} entries"
    return f"{path.stat().st_size / 1e6:.1f} MB"


def main() -> None:
    print(f"GZM_DATA = {DATA}")
    print(f"GZM_WORK = {WORK}")
    print()
    for name, path in [
        ("gz2_hart16.csv", GZ2_CATALOG),
        ("gz2_filename_mapping.csv", GZ2_MAPPING),
        ("images_gz2/", GZ2_IMAGES),
        ("Galaxy10_DECals.h5", DECALS_H5),
        ("arrays/", ARRAYS),
        ("runs/", RUNS),
        ("results/", RESULTS),
    ]:
        print(f"  {name:32s} {_describe(path)}")


if __name__ == "__main__":
    main()
