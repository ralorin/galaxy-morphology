"""Build the Galaxy10 DECaLS evaluation set.

    python -m src.prepare_decals

Galaxy10 DECaLS is used here only as a held-out survey: models are trained on
SDSS cutouts from Galaxy Zoo 2 and evaluated on DESI Legacy Imaging Surveys
cutouts without any adaptation, which is the closest thing to the situation a
survey pipeline actually faces when it is pointed at a new instrument.

Mapping the ten classes onto our binary question is not arbitrary. Galaxy10's
labels come from the same Galaxy Zoo decision tree, so the three smooth classes
are task-01 "smooth" and the five disk classes (barred, tight/loose unbarred,
edge-on with and without bulge) are task-01 "features or disk". The two remaining
classes, disturbed and merging, do not belong on either side of that question and
are dropped. This is exactly why we defined the GZ2 label from task 01 rather
than from the E/S prefix: the two surveys then answer the same question.

Output in $GZM_WORK/arrays:
    decals_table.csv   asset row index, label, original Galaxy10 class
    decals_images.npy  uint8 memmap (N, 160, 160, 3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.common import ensure_dir, timer, write_json

TABLE = config.ARRAYS / "decals_table.csv"
IMAGES = config.ARRAYS / "decals_images.npy"
META = config.ARRAYS / "decals_meta.json"

# Galaxy10 DECaLS class order as documented by astroNN.
GALAXY10_CLASSES = [
    "disturbed",                    # 0
    "merging",                      # 1
    "round smooth",                 # 2
    "in-between round smooth",      # 3
    "cigar shaped smooth",          # 4
    "barred spiral",                # 5
    "unbarred tight spiral",        # 6
    "unbarred loose spiral",        # 7
    "edge-on without bulge",        # 8
    "edge-on with bulge",           # 9
]

# 0 = smooth, 1 = featured/disk, None = not a task-01 answer, dropped
BINARY_MAP = {2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}


def main() -> None:
    import cv2
    import h5py

    if not config.DECALS_H5.exists():
        raise SystemExit(f"missing {config.DECALS_H5}; run src.download_data --only decals")

    ensure_dir(config.ARRAYS)
    size = config.CACHE_SIZE
    crop = config.DECALS_CROP

    with h5py.File(config.DECALS_H5, "r") as f:
        raw_labels = np.asarray(f["ans"]).astype(int)
        n_total = raw_labels.size
        keep = np.array([lab in BINARY_MAP for lab in raw_labels])
        idx = np.nonzero(keep)[0]
        labels = np.array([BINARY_MAP[lab] for lab in raw_labels[idx]], dtype=np.int8)

        print(f"Galaxy10 DECaLS: {n_total:,} images, keeping {idx.size:,} "
              f"({100 * labels.mean():.1f}% featured)")
        for c, name in enumerate(GALAXY10_CLASSES):
            n = int((raw_labels == c).sum())
            target = BINARY_MAP.get(c)
            tag = "dropped" if target is None else config.CLASS_NAMES[target]
            print(f"  {c} {name:26s} {n:6,d}  -> {tag}")

        arr = np.lib.format.open_memmap(
            IMAGES, mode="w+", dtype=np.uint8, shape=(idx.size, size, size, 3))
        images = f["images"]
        with timer("crop and resize"):
            # h5py fancy indexing is slow, so walk the file in contiguous blocks
            block = 512
            for start in range(0, idx.size, block):
                sel = idx[start:start + block]
                chunk = images[sel[0]:sel[-1] + 1]
                for j, global_i in enumerate(sel):
                    img = chunk[global_i - sel[0]]
                    h, w = img.shape[:2]
                    top, left = (h - crop) // 2, (w - crop) // 2
                    patch = img[top:top + crop, left:left + crop]
                    arr[start + j] = cv2.resize(patch, (size, size),
                                                interpolation=cv2.INTER_AREA)
        arr.flush()

    df = pd.DataFrame({
        "row": np.arange(idx.size, dtype=np.int64),
        "galaxy10_class": raw_labels[idx],
        "galaxy10_name": [GALAXY10_CLASSES[c] for c in raw_labels[idx]],
        "label": labels,
        "split": "test",
    })
    df.to_csv(TABLE, index=False)
    write_json(META, {
        "n_total": int(n_total),
        "n_kept": int(idx.size),
        "featured_fraction": float(labels.mean()),
        "cache_size": size,
        "crop": crop,
        "dropped_classes": ["disturbed", "merging"],
    })
    print(f"wrote {TABLE} and {IMAGES}")


if __name__ == "__main__":
    main()
