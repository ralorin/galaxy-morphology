"""Fetch the two public datasets into $GZM_DATA.

    python -m src.download_data              # everything
    python -m src.download_data --only decals

The GZ2 image archive is 3.4 GB so this is a login-node job, not something to put
inside a batch script. Downloads resume with an HTTP range request if the file is
already partially there, which matters on a flaky connection.

Sources and licences:
  * Galaxy Zoo 2 debiased catalogue, Hart et al. (2016), data.galaxyzoo.org
  * Galaxy Zoo 2 images from the original sample, Zenodo 3565489, CC BY 4.0
  * Galaxy10 DECaLS, Zenodo 10845026, images from the DESI Legacy Imaging Surveys
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import zipfile
from pathlib import Path

import requests

import config
from src.common import ensure_dir

CHUNK = 1 << 20  # 1 MiB


def download(url: str, dest: Path) -> Path:
    ensure_dir(dest.parent)
    if dest.exists():
        print(f"  {dest.name}: already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if have and r.status_code == 416:  # already complete
            part.rename(dest)
            return dest
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        done = have
        with open(part, mode) as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {dest.name}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB "
                          f"({pct:5.1f}%)", end="", flush=True)
        print()
    part.rename(dest)
    return dest


def gunzip(src: Path, dest: Path) -> Path:
    if dest.exists():
        print(f"  {dest.name}: already extracted")
        return dest
    print(f"  extracting {src.name} -> {dest.name}")
    with gzip.open(src, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout, CHUNK)
    return dest


def get_gz2_catalog() -> None:
    print("Galaxy Zoo 2 catalogue (Hart et al. 2016)")
    gz = download(config.URLS["gz2_hart16"], config.DATA / "gz2_hart16.csv.gz")
    gunzip(gz, config.GZ2_CATALOG)
    download(config.URLS["gz2_mapping"], config.GZ2_MAPPING)


def get_gz2_images() -> None:
    print("Galaxy Zoo 2 images (Zenodo 3565489, 3.4 GB)")
    archive = download(config.URLS["gz2_images"], config.DATA / "images_gz2.zip")
    marker = config.GZ2_IMAGES / ".unpacked"
    if marker.exists():
        print("  already unpacked")
        return
    print("  unpacking (this takes a while, ~240k files)")
    ensure_dir(config.GZ2_IMAGES)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(config.GZ2_IMAGES)
    marker.touch()
    # The archive nests everything under images_gz2/images/, note it so that
    # prepare_gz2 does not have to guess.
    found = list(config.GZ2_IMAGES.rglob("*.jpg"))[:1]
    if found:
        print(f"  images live in {found[0].parent}")


def get_decals() -> None:
    print("Galaxy10 DECaLS (Zenodo 10845026, 2.7 GB)")
    download(config.URLS["decals"], config.DECALS_H5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["catalog", "images", "decals"], nargs="*",
                    help="download only these pieces (default: all)")
    args = ap.parse_args()

    wanted = set(args.only or ["catalog", "images", "decals"])
    if "catalog" in wanted:
        get_gz2_catalog()
    if "images" in wanted:
        get_gz2_images()
    if "decals" in wanted:
        get_decals()
    print("\ndone. `python config.py` shows what is in place.")


if __name__ == "__main__":
    main()
