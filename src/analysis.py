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
        row.update(tuned_metrics(row["run_id"]))
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
# Operating point
# --------------------------------------------------------------------------- #

def choose_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """The threshold on the validation set that maximises balanced accuracy.

    A fixed threshold of 0.5 is only meaningful when the class prior of the
    training target matches that of the evaluation set. Two situations in this
    study break that, and both would otherwise be misread as a failure of the
    model rather than of the threshold:

      * the `*_debiased` targets are trained against fractions whose mean is far
        from the mean of the raw labels they are scored against;
      * Galaxy10 DECaLS has a substantially higher featured fraction than
        Galaxy Zoo 2, so a model transferred between the two surveys meets a
        different prior.

    We therefore report every threshold metric twice: at 0.5, which is what the
    literature reports, and at this operating point. The threshold is chosen on
    validation data only -- for the cross-survey evaluation, on the *source*
    survey's validation split -- so nothing about the test set or the target
    survey leaks into it, and the zero-shot protocol stays zero-shot.
    """
    from sklearn.metrics import balanced_accuracy_score

    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return 0.5
    # the useful candidates are the midpoints between adjacent distinct scores;
    # a few hundred quantiles capture the optimum without scanning every one
    candidates = np.unique(np.quantile(prob, np.linspace(0.001, 0.999, 400)))
    best, best_score = 0.5, -1.0
    for t in candidates:
        score = balanced_accuracy_score(y, (prob >= t).astype(int))
        if score > best_score:
            best, best_score = float(t), score
    return best


EPS = 1e-6


def _to_logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_temperature(y: np.ndarray, prob: np.ndarray) -> float:
    """The temperature that minimises validation NLL (Guo et al. 2017).

    This matters for a reason specific to this study rather than as routine
    hygiene. A model trained on the vote fraction learns to predict $p$, not
    $P(y=1 \\mid x)$: for a galaxy where three volunteers in ten said featured it
    should output about 0.3, even though the thresholded label is 0 with certainty.
    Scored against the thresholded label, such a model looks badly calibrated when
    it is in fact well calibrated for a different quantity. Temperature scaling
    separates the two: it cannot change the ranking, so whatever calibration error
    survives it is a genuine failure of the model rather than a mismatch of scale.
    """
    from scipy.optimize import minimize_scalar

    y = np.asarray(y, dtype=np.float64)
    logit = _to_logit(prob)

    def nll(log_t: float) -> float:
        p = 1.0 / (1.0 + np.exp(-logit / np.exp(log_t)))
        p = np.clip(p, EPS, 1 - EPS)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    result = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_to_logit(prob) / temperature))


def tuned_metrics(run_id: str) -> dict:
    """Metrics at the validation-chosen operating point, and after recalibration.

    Both corrections are fitted on validation data only and applied to the test
    split and to the held-out survey; neither can change the ranking of the
    predictions, so AUC is untouched by either.
    """
    val = _predictions(run_id, "val")
    test = _predictions(run_id, "test")
    if val is None or test is None:
        return {}

    threshold = choose_threshold(val["label"], val["prob"])
    temperature = fit_temperature(val["label"], val["prob"])
    out = {"val_threshold": threshold, "temperature": temperature}

    for key, frame in (("test", test), ("decals", _predictions(run_id, "decals"))):
        if frame is None:
            continue
        m = classification_metrics(frame["label"], frame["prob"], threshold=threshold)
        for name in ("accuracy", "balanced_accuracy", "f1", "mcc", "recall",
                     "precision"):
            out[f"{key}_{name}_tuned"] = m[name]
        cal = classification_metrics(frame["label"],
                                     apply_temperature(frame["prob"], temperature))
        for name in ("ece", "brier", "nll"):
            out[f"{key}_{name}_calibrated"] = cal[name]
    return out


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


def vote_ceiling(table: pd.DataFrame, column: str = "p_featured") -> dict:
    """Ceilings implied by the volunteer votes on the test split.

    The panel size is `votes_binary`, the number of volunteers who gave one of the
    two galaxy answers, because that is the panel the renormalised proportion was
    computed over.
    """
    test = table[table["split"] == "test"] if "split" in table else table
    p = test[column].to_numpy(dtype=np.float64)
    count_col = "votes_binary" if "votes_binary" in test else "votes"
    n = test[count_col].to_numpy(dtype=np.float64)
    pi = panel_probability(p, n)

    bayes = float(np.mean(np.maximum(pi, 1.0 - pi)))
    human_human = float(np.mean(pi ** 2 + (1.0 - pi) ** 2))
    label_noise = float(np.mean(np.minimum(pi, 1.0 - pi)))
    mean_ci = bootstrap_ci(np.maximum(pi, 1.0 - pi), n_boot=1000)

    return {
        "column": column,
        "panel_size_column": count_col,
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

    The class prior varies a great deal between bins on this catalogue -- the
    near-unanimous bin is mostly featured, the 0.6-0.8 bin mostly smooth -- so raw
    accuracy is not comparable across bins on its own. We therefore also record the
    within-bin majority baseline and the balanced accuracy, and the figure plots the
    baseline alongside the curve. `lift` is accuracy minus that baseline, which is
    the honest quantity to compare between bins.
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
        total_errors = float((((pred[col] >= 0.5).astype(int) != pred["label"]).sum()))
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() < 20:
                continue
            y = pred.loc[m, "label"]
            metrics = classification_metrics(y, pred.loc[m, col])
            prior = float(y.mean())
            baseline = max(prior, 1.0 - prior)
            errors = float(((pred.loc[m, col] >= 0.5).astype(int) != y).sum())
            rows.append({
                "run_id": run_id,
                "bin": b,
                "agreement_low": edges[b],
                "agreement_high": edges[b + 1],
                "n": int(m.sum()),
                "share": float(m.mean()),
                "featured_fraction": prior,
                "majority_baseline": baseline,
                "lift": metrics["accuracy"] - baseline,
                # what fraction of the model's total errors this bin accounts for
                "error_share": errors / total_errors if total_errors else 0.0,
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


def calibration_curves(runs: pd.DataFrame, run_ids: list[str], n_bins: int = 15
                       ) -> pd.DataFrame:
    """Reliability curves, one row per (run, bin).

    The `bin` index is what to average over when pooling runs: `confidence` is the
    mean predicted probability inside the bin and therefore differs slightly from
    one run to the next, so grouping on it produces a ragged curve rather than a
    pooled one.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for run_id in run_ids:
        pred = _predictions(run_id)
        if pred is None:
            continue
        centres, observed, counts = reliability_curve(pred["label"], pred["prob"],
                                                      n_bins=n_bins)
        for b, (c, o, n) in enumerate(zip(centres, observed, counts)):
            rows.append({"run_id": run_id, "bin": b,
                         "bin_centre": float((edges[b] + edges[b + 1]) / 2),
                         "confidence": c, "observed": o, "count": int(n)})
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
    # The prior differs by tens of points between the two surveys, so the drop at a
    # fixed 0.5 threshold conflates a shift in prior with a loss of discrimination.
    # These two separate them: the AUC drop is threshold-free, and the tuned drop
    # uses the operating point chosen on the source survey's validation split.
    if "decals_balanced_accuracy_tuned" in have and "test_balanced_accuracy_tuned" in have:
        have["balanced_accuracy_drop_tuned"] = (have["test_balanced_accuracy_tuned"]
                                                - have["decals_balanced_accuracy_tuned"])
    cols = ["run_id", *CONFIG_KEYS, "val_threshold",
            "test_accuracy", "decals_accuracy", "accuracy_drop",
            "test_auc_roc", "decals_auc_roc", "auc_drop",
            "test_balanced_accuracy", "decals_balanced_accuracy",
            "test_balanced_accuracy_tuned", "decals_balanced_accuracy_tuned",
            "balanced_accuracy_drop_tuned", "decals_tta_accuracy"]
    return have[[c for c in cols if c in have]]


# --------------------------------------------------------------------------- #

def learning_curve_reach(runs: pd.DataFrame, ceiling: dict,
                         train_size: int | None = None) -> dict:
    """How much more annotation the learning curves say the ceiling would cost.

    The honest reading of the curves is not that they have flattened -- they have
    not -- but that their slope is shallow enough for the remaining gap to be
    unaffordable. Accuracy grows roughly linearly in the logarithm of the training
    set size, so we fit that slope over the last three points of each curve and ask
    how many decades of additional data the gap to the vote-model ceiling implies.
    The answer is the quantity worth reporting: not "more data will not help" but
    "the data that would help does not exist".

    Fitting the last three points rather than all of them matters. The early part of
    a learning curve is steeper, and including it would flatter the extrapolation.
    """
    if train_size is None:
        meta = config.ARRAYS / "gz2_meta.json"
        train_size = int(read_json(meta)["split_counts"]["train"]) \
            if meta.exists() else 0

    # accept either the raw-fraction block or the whole ceiling record, since callers
    # hold the latter; with the wrong one this silently returns nothing at all
    if "bayes_accuracy" not in ceiling and "raw" in ceiling:
        ceiling = ceiling["raw"]
    target = ceiling.get("bayes_accuracy")
    sub = runs[(runs["policy"] == "d4") & (runs["finetune"] == "full")
               & (runs["loss"] == "bce")
               & (~runs["orientation_pooled"].astype(bool))
               & (runs["label_mode"].isin(["hard", "soft"]))]
    if "pretrained" in sub:
        sub = sub[sub["pretrained"].fillna(True).astype(bool)]
    # train_size == 0 is the sentinel for "the whole training split", so it has to be
    # resolved to the real count. If it cannot be, the full-split runs must be dropped
    # rather than left at zero: log10(0) would silently move the last three points of
    # every curve to the subset runs and return a slope and a gap that look entirely
    # reasonable and are wrong.
    if not train_size:
        print("learning_curve_reach: no training-set count available, so the "
              "full-split runs are excluded from the curves")
        sub = sub[sub["train_size"] > 0]
    sub = sub.assign(n=sub["train_size"].replace(0, train_size))

    curves = {}
    for (arch, mode), group in sub.groupby(["arch", "label_mode"]):
        m = group.groupby("n")["test_accuracy"].mean().sort_index()
        if len(m) < 3 or target is None:
            continue
        x = np.log10(m.index.to_numpy(dtype=float))
        y = m.to_numpy()
        slope = float(np.polyfit(x[-3:], y[-3:], 1)[0])
        gap = float(target - y[-1])
        decades = gap / slope if slope > 0 else float("inf")
        curves[f"{arch}/{mode}"] = {
            "final_accuracy": float(y[-1]),
            "gap_to_ceiling": gap,
            "slope_per_decade": slope,
            "decades_needed": decades,
            "galaxies_needed": float(m.index[-1] * 10 ** decades),
            "still_rising": bool(slope > 0.002),
        }

    if not curves:
        return {}
    finite = [c for c in curves.values() if np.isfinite(c["decades_needed"])]
    return {
        "per_curve": curves,
        "n_curves": len(curves),
        "n_still_rising": int(sum(c["still_rising"] for c in curves.values())),
        "mean_slope_per_decade": float(np.mean([c["slope_per_decade"]
                                               for c in curves.values()])),
        "mean_gap_to_ceiling": float(np.mean([c["gap_to_ceiling"]
                                              for c in curves.values()])),
        "galaxies_needed_min": float(min(c["galaxies_needed"] for c in finite)) if finite else None,
        "galaxies_needed_max": float(max(c["galaxies_needed"] for c in finite)) if finite else None,
    }


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
        "learning_curve": learning_curve_reach(runs, ceiling),
    }

    if not agreement.empty:
        top = agreement[agreement["label_mode"] == "soft"]
        cols = ["accuracy", "balanced_accuracy", "majority_baseline", "lift",
                "error_share", "share", "n"]
        by_bin = top.groupby("bin")[[c for c in cols if c in top]].mean()
        out["accuracy_by_agreement_bin"] = {
            f"{config.AGREEMENT_BINS[int(b)]:.1f}-{config.AGREEMENT_BINS[int(b) + 1]:.1f}":
                {k: float(v) for k, v in r.items()}
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
    ceiling = vote_ceiling(table, "p_featured")
    ceiling_debiased = vote_ceiling(table, "p_featured_debiased") \
        if "p_featured_debiased" in table else {}
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
