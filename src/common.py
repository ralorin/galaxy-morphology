"""Small helpers shared by the rest of the pipeline: seeding, device selection,
json/npz io and the metric functions we report.

Nothing here is specific to galaxies; it is the plumbing that keeps the other
modules short.
"""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

try:  # torch is not needed for the pure-analysis scripts
    import torch
except ImportError:  # pragma: no cover
    torch = None


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch.

    `deterministic=True` also forces cuDNN into its deterministic algorithms.
    We leave it off for the training runs because it costs roughly 20-30% of the
    throughput and we average over five seeds anyway; it is switched on for the
    small evaluation and explanation jobs where speed does not matter.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True


def get_device() -> "torch.device":
    if torch is None:
        raise RuntimeError("torch is not installed")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device() -> str:
    if torch is None or not torch.cuda.is_available():
        return "cpu"
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return f"{props.name} ({props.total_memory / 1e9:.0f} GB, sm_{props.major}{props.minor})"


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value)}")


def load_table(which: str = "gz2"):
    """The label table produced by the prepare scripts, as a DataFrame.

    Lives here rather than in `datasets` so that the analysis and figure scripts
    can read it without importing torch.
    """
    import pandas as pd

    import config

    path = config.ARRAYS / f"{which}_table.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run src.prepare_{which} first")
    return pd.read_csv(path)


def dataset_meta() -> dict:
    """The record prepare_gz2 wrote, wherever it happens to be.

    It sits next to the arrays on the cluster and is copied into results/ when the
    results are published, so anyone working from a clone finds it in the second
    place and not the first. Looking in one place only was silently costing the
    learning-curve figure the true size of the training split.
    """
    import config

    for path in (config.ARRAYS / "gz2_meta.json", config.RESULTS / "gz2_meta.json"):
        if path.exists():
            return read_json(path)
    return {}


def train_split_size() -> int:
    counts = dataset_meta().get("split_counts", {})
    if "train" not in counts:
        raise SystemExit("no training-set size available: gz2_meta.json is missing "
                         "from both the arrays and the results")
    return int(counts["train"])


@contextmanager
def timer(label: str = ""):
    """Wall-clock timer. `with timer() as t: ...` then read `t.seconds`."""
    class _T:
        seconds = 0.0

    t = _T()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.seconds = time.perf_counter() - start
        if label:
            print(f"[{label}] {t.seconds:.1f} s", flush=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Threshold metrics plus the two ranking metrics and the calibration ones.

    `y_true` is the hard 0/1 label, `prob` the predicted probability of class 1
    (featured). Positive class = featured, which is the majority class here; the
    balanced accuracy and MCC below are what to look at when the class ratio
    changes between datasets.
    """
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        matthews_corrcoef,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    out = {
        "n": int(y_true.size),
        "accuracy": float((tp + tn) / y_true.size),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(y_true)) > 1:
        out["auc_roc"] = float(roc_auc_score(y_true, prob))
        out["auc_pr"] = float(average_precision_score(y_true, prob))
    else:  # a stratified bin can end up single-class
        out["auc_roc"] = float("nan")
        out["auc_pr"] = float("nan")
    out["brier"] = float(brier_score_loss(y_true, prob))
    out["ece"] = float(expected_calibration_error(y_true, prob))
    out["nll"] = float(negative_log_likelihood(y_true, prob))
    return out


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width binning ECE, the usual definition (Guo et al. 2017)."""
    y_true = np.asarray(y_true).astype(float)
    prob = np.asarray(prob, dtype=np.float64)
    conf = np.where(prob >= 0.5, prob, 1.0 - prob)
    correct = ((prob >= 0.5).astype(float) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def negative_log_likelihood(y_true: np.ndarray, prob: np.ndarray, eps: float = 1e-7) -> float:
    y_true = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(prob, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def reliability_curve(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15):
    """Confidence vs accuracy per bin, for the reliability diagrams."""
    y_true = np.asarray(y_true).astype(float)
    prob = np.asarray(prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    centres, observed, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        counts.append(int(m.sum()))
        if m.any():
            centres.append(float(prob[m].mean()))
            observed.append(float(y_true[m].mean()))
        else:
            centres.append(float((edges[b] + edges[b + 1]) / 2))
            observed.append(float("nan"))
    return np.array(centres), np.array(observed), np.array(counts)


def risk_coverage_curve(y_true: np.ndarray, prob: np.ndarray, confidence: np.ndarray | None = None):
    """Selective prediction curve.

    Sort the test set by confidence, then walk down it reporting the accuracy of
    the most confident fraction. `confidence` defaults to max(p, 1-p). Returns
    (coverage, accuracy, risk) with coverage from 1/n to 1.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    if confidence is None:
        confidence = np.maximum(prob, 1.0 - prob)
    correct = ((prob >= 0.5).astype(int) == y_true).astype(np.float64)

    order = np.argsort(-np.asarray(confidence, dtype=np.float64), kind="stable")
    correct = correct[order]
    n = correct.size
    cumulative = np.cumsum(correct)
    coverage = np.arange(1, n + 1) / n
    accuracy = cumulative / np.arange(1, n + 1)
    return coverage, accuracy, 1.0 - accuracy


def coverage_at_accuracy(coverage: np.ndarray, accuracy: np.ndarray, target: float) -> float:
    """Largest coverage whose selective accuracy is still at or above `target`.

    Returns 0.0 if the target is never reached. The curve is not monotone, so we
    take the last crossing rather than the first.
    """
    ok = np.nonzero(accuracy >= target)[0]
    if ok.size == 0:
        return 0.0
    return float(coverage[ok[-1]])


def bootstrap_ci(values: np.ndarray, statistic=np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI over the rows of `values`."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = values.shape[0]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = statistic(values[rng.integers(0, n, n)])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(statistic(values)), float(lo), float(hi)
