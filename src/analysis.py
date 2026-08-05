"""Turn the per-run outputs into the tables the paper is built on.

    python -m src.analysis

Writes into $GZM_WORK/results:

    runs.csv              one tidy row per run: configuration plus every metric
    agreement.csv         accuracy and calibration inside each volunteer-agreement bin
    ceiling.json          the vote-model ceiling, computed from the catalogue alone
    selective.csv         risk-coverage summaries, coverage at 95/98/99% accuracy
    calibration.csv       reliability curves for the runs we plot
    cross_survey.csv      SDSS -> DECaLS transfer for every run that has it
    summary.json          the handful of numbers quoted in the abstract

The one piece of modelling in here is `vote_ceiling`, so it is worth being explicit
about what it does and what it bounds.

A Galaxy Zoo label is a threshold applied to a *measurement*. What the catalogue
records is not the volunteer pool's propensity p* to call a galaxy featured but an
estimate of it, p_hat = K/N, from the N volunteers who happened to be asked. The
label everybody trains and tests on is 1[p_hat > 0.5]. Two galaxies identical in
appearance therefore share a propensity but can carry different recorded labels,
purely because different people looked at them.

Model the votes as Bernoulli draws with rate p and panel size N. The probability
that a panel returns "featured" is

    pi(p, N) = P(Bin(N, p) > N/2) + 0.5 * P(Bin(N, p) = N/2)

so P(y = 1 | image) = pi(p*, N) and the Bayes-optimal predictor of the *recorded*
label attains E[max(pi, 1 - pi)]. That is a bound on accuracy against the frozen
test labels, not merely against some hypothetical fresh panel: a classifier sees
pixels, and no function of the pixels can anticipate which way one particular
panel fell.

The estimation step biases the result in a convenient direction. We substitute
p_hat for p*, and max(pi, 1-pi) is convex, so by Jensen the plug-in expectation
*over*-estimates the bound. The ceiling reported here is therefore generous to the
classifiers rather than harsh: an accuracy below it says nothing on its own, while
an accuracy above it would be evidence against the model or against the
independence of the test split.

The assumptions are worth stating too, and the paper states them. Volunteers are
not exchangeable; the debiasing of Hart et al. means the published fraction is not
a raw proportion, so we feed the raw fractions in and report the debiased variant
as a sensitivity check; and the binomial ignores correlations in who classified
what.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.common import (bootstrap_ci, classification_metrics, coverage_at_accuracy,
                        ensure_dir, load_table, read_json, reliability_curve,
                        risk_coverage_curve, write_json)

CONFIG_KEYS = ("arch", "label_mode", "policy", "size", "finetune", "loss",
               "train_size", "seed", "orientation_pooled", "pretrained")
METRIC_KEYS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc",
               "auc_roc", "auc_pr", "brier", "ece", "nll")

# numpy renamed trapz in 2.0; we pin 1.26 but do not want to break on a newer one
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
# Collecting
# --------------------------------------------------------------------------- #

def collect_runs() -> pd.DataFrame:
    rows = []
    for path in sorted(config.RUNS.glob("*/metrics.json")):
        m = read_json(path)
        cfg = m.get("config", {})
        row = {"run_id": m.get("run_id", path.parent.name)}
        row.update({k: cfg.get(k) for k in CONFIG_KEYS})
        row["pretrained"] = cfg.get("pretrained", True)
        row["train_size"] = cfg.get("train_size") or 0     # 0 means "the whole split"
        row["epochs_run"] = m.get("epochs_run")
        row["best_epoch"] = m.get("best_epoch")
        row["train_seconds"] = m.get("train_seconds")
        row["images_per_second"] = m.get("images_per_second")
        row["params"] = m.get("params")
        row["params_trainable"] = m.get("params_trainable")
        row["d4_invariance_error"] = m.get("d4_invariance_error")
        for split in ("test", "test_tta", "val", "decals", "decals_tta"):
            block = m.get(split)
            if isinstance(block, dict):
                for k in METRIC_KEYS:
                    row[f"{split}_{k}"] = block.get(k)
        rows.append(row)

    if not rows:
        raise SystemExit(f"no runs found under {config.RUNS}")
    df = pd.DataFrame(rows)
    df["orientation_pooled"] = df["orientation_pooled"].fillna(False).astype(bool)
    if "pretrained" in df:
        df["pretrained"] = df["pretrained"].fillna(True).astype(bool)
    print(f"collected {len(df)} runs "
          f"({df['arch'].nunique()} architectures, {df['seed'].nunique()} seeds)")
    return df


def _predictions(run_id: str, which: str = "test") -> pd.DataFrame | None:
    path = config.RUNS / run_id / f"predictions_{which}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# The vote model
# --------------------------------------------------------------------------- #

def panel_probability(p: np.ndarray, n: np.ndarray) -> np.ndarray:
    """P(a panel of n volunteers returns 'featured'), ties split evenly."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    n = np.maximum(np.asarray(n, dtype=np.int64), 1)
    half = n / 2.0
    # P(X > n/2) = 1 - cdf(floor(n/2)), and the tie term only exists for even n
    strictly_more = 1.0 - stats.binom.cdf(np.floor(half), n, p)
    tie = np.where(n % 2 == 0, stats.binom.pmf(n // 2, n, p), 0.0)
    return np.clip(strictly_more + 0.5 * tie, 0.0, 1.0)


def vote_ceiling(table: pd.DataFrame, column: str = "p_featured_raw") -> dict:
    """Ceilings implied by the volunteer votes on the test split."""
    test = table[table["split"] == "test"] if "split" in table else table
    p = test[column].to_numpy(dtype=np.float64)
    n = test["votes"].to_numpy(dtype=np.float64)
    pi = panel_probability(p, n)

    bayes = float(np.mean(np.maximum(pi, 1.0 - pi)))
    human_human = float(np.mean(pi ** 2 + (1.0 - pi) ** 2))
    label_noise = float(np.mean(np.minimum(pi, 1.0 - pi)))
    mean_ci = bootstrap_ci(np.maximum(pi, 1.0 - pi), n_boot=1000)

    return {
        "column": column,
        "n_test": int(test.shape[0]),
        "median_votes": float(np.median(n)),
        # best possible accuracy against a freshly drawn panel label
        "bayes_accuracy": bayes,
        "bayes_accuracy_ci": [mean_ci[1], mean_ci[2]],
        # how often two independent panels would agree with each other
        "panel_agreement": human_human,
        # how often the recorded label is the minority answer of a fresh panel
        "label_noise_rate": label_noise,
    }


# --------------------------------------------------------------------------- #
# Agreement-conditioned behaviour
# --------------------------------------------------------------------------- #

def agreement_profile(runs: pd.DataFrame, prob_column: str = "prob") -> pd.DataFrame:
    """Accuracy, AUC and calibration inside each volunteer-agreement bin.

    This is the central measurement of the paper: if the residual error is
    concentrated where the volunteers disagreed, the ceiling is a property of the
    labels; if it is spread evenly, it is a property of the models.
    """
    edges = np.array(config.AGREEMENT_BINS)
    rows = []
    for run_id in runs["run_id"]:
        pred = _predictions(run_id)
        if pred is None:
            continue
        col = prob_column if prob_column in pred else "prob"
        idx = np.clip(np.digitize(pred["agreement"].to_numpy(), edges[1:-1]), 0,
                      len(edges) - 2)
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() < 20:
                continue
            metrics = classification_metrics(pred.loc[m, "label"], pred.loc[m, col])
            rows.append({
                "run_id": run_id,
                "bin": b,
                "agreement_low": edges[b],
                "agreement_high": edges[b + 1],
                "n": int(m.sum()),
                "share": float(m.mean()),
                **{k: metrics[k] for k in ("accuracy", "balanced_accuracy", "auc_roc",
                                           "ece", "brier", "n")},
            })
    out = pd.DataFrame(rows)
    return out.merge(runs[["run_id", *CONFIG_KEYS]], on="run_id", how="left")


def probability_vs_votes(runs: pd.DataFrame) -> pd.DataFrame:
    """Does the predicted probability track the human vote fraction?

    Spearman correlation and mean absolute error between the network's output and
    the debiased fraction. A model trained on thresholded labels has no reason to
    reproduce the fraction; one trained on it should, and if it does its confidence
    is directly interpretable as expected human disagreement.
    """
    rows = []
    for run_id in runs["run_id"]:
        pred = _predictions(run_id)
        if pred is None:
            continue
        p_hat = pred["prob"].to_numpy()
        p_true = pred["p_featured"].to_numpy()
        rho, _ = stats.spearmanr(p_hat, p_true)
        rows.append({
            "run_id": run_id,
            "spearman_vs_votefraction": float(rho),
            "mae_vs_votefraction": float(np.mean(np.abs(p_hat - p_true))),
            "pearson_vs_votefraction": float(np.corrcoef(p_hat, p_true)[0, 1]),
        })
    out = pd.DataFrame(rows)
    return out.merge(runs[["run_id", *CONFIG_KEYS]], on="run_id", how="left")


# --------------------------------------------------------------------------- #
# Selective prediction
# --------------------------------------------------------------------------- #

def selective_prediction(runs: pd.DataFrame,
                         targets=(0.95, 0.98, 0.99)) -> pd.DataFrame:
    """Coverage reachable at a set of accuracy targets, per run.

    The operational reading: a survey pipeline that is willing to send the least
    confident x% of its objects to a human reviewer can hold the automated part at
    a chosen accuracy. `coverage_at_99` is how much of the sky the network can be
    trusted with if the requirement is 99% correct.
    """
    rows = []
    for run_id in runs["run_id"]:
        pred = _predictions(run_id)
        if pred is None:
            continue
        cov, acc, _ = risk_coverage_curve(pred["label"], pred["prob"])
        entry = {"run_id": run_id,
                 "full_accuracy": float(acc[-1]),
                 # area under the accuracy-coverage curve, one number for the whole
                 # trade-off; higher is better
                 "auc_coverage": float(_trapezoid(acc, cov))}
        for t in targets:
            entry[f"coverage_at_{int(t * 100)}"] = coverage_at_accuracy(cov, acc, t)
        rows.append(entry)
    out = pd.DataFrame(rows)
    return out.merge(runs[["run_id", *CONFIG_KEYS]], on="run_id", how="left")


def calibration_curves(runs: pd.DataFrame, run_ids: list[str]) -> pd.DataFrame:
    rows = []
    for run_id in run_ids:
        pred = _predictions(run_id)
        if pred is None:
            continue
        centres, observed, counts = reliability_curve(pred["label"], pred["prob"])
        for c, o, n in zip(centres, observed, counts):
            rows.append({"run_id": run_id, "confidence": c, "observed": o, "count": int(n)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.merge(runs[["run_id", *CONFIG_KEYS]], on="run_id", how="left")


# --------------------------------------------------------------------------- #
# Cross-survey
# --------------------------------------------------------------------------- #

def cross_survey(runs: pd.DataFrame) -> pd.DataFrame:
    have = runs[runs["decals_accuracy"].notna()].copy()         if "decals_accuracy" in runs else pd.DataFrame()
    if have.empty:
        print("no cross-survey results yet (run slurm/04_cross_survey.sh)")
        return have
    have["accuracy_drop"] = have["test_accuracy"] - have["decals_accuracy"]
    have["auc_drop"] = have["test_auc_roc"] - have["decals_auc_roc"]
    cols = ["run_id", *CONFIG_KEYS, "test_accuracy", "decals_accuracy", "accuracy_drop",
            "test_auc_roc", "decals_auc_roc", "auc_drop",
            "decals_tta_accuracy", "decals_balanced_accuracy"]
    return have[[c for c in cols if c in have]]


# --------------------------------------------------------------------------- #

def summarise(runs: pd.DataFrame, agreement: pd.DataFrame, ceiling: dict,
              selective: pd.DataFrame) -> dict:
    """The numbers that go in the abstract, pulled from the tables above."""
    ref = runs[(runs["policy"] == "d4") & (runs["finetune"] == "full")
               & (runs["train_size"] == 0) & (~runs["orientation_pooled"].astype(bool))]

    def best(frame, mode):
        sub = frame[frame["label_mode"] == mode]
        if sub.empty:
            return {}
        per_arch = sub.groupby("arch")["test_accuracy"].mean().sort_values()
        arch = per_arch.index[-1]
        rows = sub[sub["arch"] == arch]
        return {
            "arch": arch,
            "accuracy_mean": float(rows["test_accuracy"].mean()),
            "accuracy_std": float(rows["test_accuracy"].std(ddof=1)) if len(rows) > 1 else 0.0,
            "auc_mean": float(rows["test_auc_roc"].mean()),
            "ece_mean": float(rows["test_ece"].mean()),
            "n_seeds": int(len(rows)),
        }

    out = {
        "n_runs": int(len(runs)),
        "vote_ceiling": ceiling,
        "best_hard": best(ref, "hard"),
        "best_soft": best(ref, "soft"),
    }

    if not agreement.empty:
        top = agreement[agreement["label_mode"] == "soft"]
        by_bin = top.groupby("bin")[["accuracy", "share", "n"]].mean()
        out["accuracy_by_agreement_bin"] = {
            f"{config.AGREEMENT_BINS[int(b)]:.1f}-{config.AGREEMENT_BINS[int(b) + 1]:.1f}":
                {"accuracy": float(r["accuracy"]), "share": float(r["share"])}
            for b, r in by_bin.iterrows()
        }
    if not selective.empty:
        sub = selective[selective["label_mode"] == "soft"]
        if not sub.empty:
            out["coverage_at_99_best"] = float(sub["coverage_at_99"].max())
            out["coverage_at_99_median"] = float(sub["coverage_at_99"].median())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-predictions", action="store_true",
                    help="only rebuild runs.csv, do not touch the per-galaxy analyses")
    args = ap.parse_args()

    ensure_dir(config.RESULTS)
    runs = collect_runs()
    runs.to_csv(config.RESULTS / "runs.csv", index=False)

    table = load_table("gz2")
    ceiling = vote_ceiling(table, "p_featured_raw")
    ceiling_debiased = vote_ceiling(table, "p_featured")
    write_json(config.RESULTS / "ceiling.json",
               {"raw": ceiling, "debiased_sensitivity": ceiling_debiased})
    print(f"vote-model ceiling: best possible accuracy against a fresh panel = "
          f"{ceiling['bayes_accuracy']:.4f}; two panels agree "
          f"{ceiling['panel_agreement']:.4f} of the time")

    agreement = selective = pd.DataFrame()
    if not args.skip_predictions:
        agreement = agreement_profile(runs)
        agreement.to_csv(config.RESULTS / "agreement.csv", index=False)

        selective = selective_prediction(runs)
        selective.to_csv(config.RESULTS / "selective.csv", index=False)

        probability_vs_votes(runs).to_csv(config.RESULTS / "vote_tracking.csv", index=False)

        # reliability curves only for the reference runs of seed 0, which is what
        # the figure shows
        wanted = runs[(runs["seed"] == 0) & (runs["policy"] == "d4")
                      & (runs["train_size"] == 0)]["run_id"].tolist()
        calibration_curves(runs, wanted).to_csv(config.RESULTS / "calibration.csv",
                                                index=False)

    cross = cross_survey(runs)
    if not cross.empty:
        cross.to_csv(config.RESULTS / "cross_survey.csv", index=False)

    summary = summarise(runs, agreement, ceiling, selective)
    write_json(config.RESULTS / "summary.json", summary)
    print(f"wrote the analysis tables to {config.RESULTS}")


if __name__ == "__main__":
    main()
