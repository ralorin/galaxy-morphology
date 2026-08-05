"""Datasets and the augmentation policies used in the orientation ablation.

The images sit in a uint8 memmap produced by `prepare_gz2` / `prepare_decals`, so
loading a sample is a slice out of a page-cached array and the augmentation is the
only real work in the worker processes.

Augmentation policies
---------------------
none      resize to the input size, nothing else
flip      random horizontal and vertical flips
d4        a uniformly drawn element of the dihedral group D4 (four rotations by
          multiples of 90 degrees, optionally mirrored: eight transforms in all)
d4_photo  d4 plus a small shift and zoom and a brightness/contrast jitter

The point of the d4 policy is that galaxy images have no canonical orientation:
the camera angle is arbitrary, so the label is invariant to the whole group. Any
accuracy a network gains from `d4` over `none` is accuracy it was losing to an
inductive bias it should never have needed. `d4_photo` adds the nuisances that do
vary between surveys (seeing, sky background, zero point) and is the policy we use
for the cross-survey runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config
from src.experiment import POLICIES

# The eight elements of D4, as (number of 90-degree rotations, mirror flag).
D4_ELEMENTS = tuple((k, m) for m in (False, True) for k in range(4))


def apply_d4(img: np.ndarray, k: int, mirror: bool) -> np.ndarray:
    out = np.rot90(img, k=k, axes=(0, 1))
    if mirror:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


@dataclass
class Normalisation:
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class GalaxyDataset(Dataset):
    """Rows of a table plus the matching rows of an image memmap.

    Each item is a dict so that the training loop can pick what it needs without
    caring about tuple order:

        image      float tensor (3, S, S), normalised
        target     float, the vote fraction (soft label)
        label      long, the thresholded label
        agreement  float, |2p-1|
        weight     float, per-sample loss weight
        row        long, index into the table (used to line up predictions)
    """

    def __init__(self, table: pd.DataFrame, images_path, size: int,
                 policy: str = "none", norm: Normalisation | None = None,
                 train: bool = True, weights: np.ndarray | None = None):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}, expected one of {POLICIES}")
        self.table = table.reset_index(drop=True)
        self.images_path = str(images_path)
        self._images = None  # opened lazily, once per worker process
        self.size = size
        self.policy = policy
        self.train = train
        self.norm = norm or Normalisation()

        self._rng = None
        self.rows = self.table["row"].to_numpy(dtype=np.int64)
        self.targets = self.table["p_featured"].to_numpy(dtype=np.float32) \
            if "p_featured" in self.table else self.table["label"].to_numpy(dtype=np.float32)
        self.labels = self.table["label"].to_numpy(dtype=np.int64)
        self.agreement = self.table["agreement"].to_numpy(dtype=np.float32) \
            if "agreement" in self.table else np.ones(len(self.table), dtype=np.float32)
        self.weights = (np.ones(len(self.table), dtype=np.float32)
                        if weights is None else weights.astype(np.float32))

    # -- image access ------------------------------------------------------- #

    @property
    def images(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.images_path, mmap_mode="r")
        return self._images

    @property
    def rng(self) -> np.random.Generator:
        """One stateful generator per worker process.

        It is seeded from the worker's torch seed, which derives from the global
        seed, so a run is reproducible for a given seed and worker count. Being
        stateful rather than re-seeded per item matters: with
        `persistent_workers=True` the workers survive between epochs, so a
        per-item seed would hand every galaxy the same augmentation every epoch.
        """
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 63))
        return self._rng

    def __len__(self) -> int:
        return len(self.table)

    # -- augmentation ------------------------------------------------------- #

    def _augment(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.policy == "none":
            return img
        if self.policy == "flip":
            if rng.random() < 0.5:
                img = img[:, ::-1]
            if rng.random() < 0.5:
                img = img[::-1, :]
            return np.ascontiguousarray(img)

        k, mirror = D4_ELEMENTS[rng.integers(len(D4_ELEMENTS))]
        img = apply_d4(img, k, mirror)

        if self.policy == "d4_photo":
            img = self._photometric(img, rng)
        return img

    def _photometric(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        # up to +-6% shift and +-10% zoom; the galaxy is centred by construction so
        # anything larger starts pushing it out of frame
        zoom = float(rng.uniform(0.90, 1.10))
        dx = float(rng.uniform(-0.06, 0.06) * w)
        dy = float(rng.uniform(-0.06, 0.06) * h)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, zoom)
        m[0, 2] += dx
        m[1, 2] += dy
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

        # brightness and contrast, in the [0,255] domain
        gain = float(rng.uniform(0.9, 1.1))
        bias = float(rng.uniform(-10, 10))
        img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        return img

    # -- item --------------------------------------------------------------- #

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        import cv2

        if img.shape[0] != self.size:
            interp = cv2.INTER_AREA if img.shape[0] > self.size else cv2.INTER_LINEAR
            img = cv2.resize(img, (self.size, self.size), interpolation=interp)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor(self.norm.mean).view(3, 1, 1)
        std = torch.tensor(self.norm.std).view(3, 1, 1)
        return (x - mean) / std

    def __getitem__(self, i: int) -> dict:
        img = np.asarray(self.images[self.rows[i]])
        if self.train:
            img = self._augment(img, self.rng)
        return {
            "image": self._to_tensor(img),
            "target": torch.tensor(self.targets[i], dtype=torch.float32),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
            "agreement": torch.tensor(self.agreement[i], dtype=torch.float32),
            "weight": torch.tensor(self.weights[i], dtype=torch.float32),
            "row": torch.tensor(self.rows[i], dtype=torch.long),
        }

    def d4_views(self, i: int) -> torch.Tensor:
        """The eight D4 transforms of one image, stacked. Used for test-time
        averaging and for the invariance measurements."""
        img = np.asarray(self.images[self.rows[i]])
        return torch.stack([self._to_tensor(apply_d4(img, k, m)) for k, m in D4_ELEMENTS])


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #

def load_gz2_table() -> pd.DataFrame:
    from src.common import load_table

    return load_table("gz2")


def load_decals_table() -> pd.DataFrame:
    from src.common import load_table

    return load_table("decals")


def subsample_train(table: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Take a class-stratified subset of the training split.

    Used for the learning curves. Stratifying on the label keeps the class ratio
    of the full set, so the curve measures the effect of size and nothing else.
    """
    if n is None or n >= len(table):
        return table
    rng = np.random.default_rng(seed)
    parts = []
    for label, group in table.groupby("label"):
        take = int(round(n * len(group) / len(table)))
        parts.append(group.iloc[rng.permutation(len(group))[:take]])
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_loader(dataset: GalaxyDataset, batch_size: int, shuffle: bool,
                workers: int, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, drop_last=drop_last,
        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
        worker_init_fn=_worker_init if workers > 0 else None,
    )


def _worker_init(worker_id: int) -> None:
    # OpenCV spawns its own thread pool per process; with eight dataloader workers
    # that oversubscribes the cores and slows everything down.
    import cv2

    cv2.setNumThreads(0)
