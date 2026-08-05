"""Significance testing over the runs.

    python -m src.stats

Four things, written to $GZM_WORK/results:

    bootstrap.csv    percentile bootstrap CIs on the test metrics of every
                     configuration, resampling the test set rather than the seeds
    mcnemar.csv      paired McNemar tests between the configurations we claim a
                     difference for, on the identical test set
    architecture_pairwise.csv
                     Holm-corrected pairwise McNemar between architectures, on the
                     seed-averaged predictions
    friedman.json    Friedman test across architectures with the per-seed
                     accuracies as blocks, plus Nemenyi post-hoc ranks
    wilcoxon.csv     paired Wilcoxon signed-rank tests for the ablation knobs
                     (hard vs soft labels, none vs d4, and so on)

Two sources of variation get confused in this kind of study: the finite test set
and the training seed. We keep them apart. The bootstrap CIs answer "would another
sample of galaxies have given the same number"; McNemar answers "do these two
models disagree on this test set more than chance would explain"; the Friedman and
Wilcoxon tests use the seeds as the unit of replication and answer "is this an
effect of the configuration or of the initialisation".
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.common import ensure_dir, write_json

ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Bootstrap over the test set
# --------------------------------------------------------------------------- #

def bootstrap_metrics(run_id: str, n_boot: int = 2000, seed: int = 0) -> dict | None:
    """Resample the test predictions and report CIs on accuracy, AUC and ECE."""
    from src.common import classification_metrics

    path = config.RUNS / run_id / "predictions_test.csv"
    if not path.exists():
        return None
    pred = pd.read_csv(path)
    y = pred["label"].to_numpy()
    p = pred["prob"].to_numpy()
    rng = np.random.default_rng(seed)

    keys = ("accuracy", "balanced_accuracy", "auc_roc", "auc_pr", "f1", "ece", "brier")
    draws = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        m = classification_metrics(y[idx], p[idx])
        for k in keys:
            draws[k][b] = m[k]

    point = classification_metrics(y, p)
    out = {"run_id": run_id, "n_test": int(y.size)}
    for k in keys:
        lo, hi = np.quantile(draws[k], [ALPHA / 2, 1 - ALPHA / 2])
        out[k] = point[k]
        out[f"{k}_lo"] = float(lo)
        out[f"{k}_hi"] = float(hi)
    return out


# --------------------------------------------------------------------------- #
# McNemar
# --------------------------------------------------------------------------- #

def mcnemar(run_a: str, run_b: str) -> dict | None:
    """Exact McNemar on the shared test set.

    The two runs are evaluated on the same galaxies in the same order, so the
    discordant pairs are well defined. We use the exact binomial version rather
    than the chi-square approximation because some of the discordant counts are
    small.
    """
    pa = config.RUNS / run_a / "predictions_test.csv"
    pb = config.RUNS / run_b / "predictions_test.csv"
    if not (pa.exists() and pb.exists()):
        return None
    a = pd.read_csv(pa).sort_values("row").reset_index(drop=True)
    b = pd.read_csv(pb).sort_values("row").reset_index(drop=True)
    if not np.array_equal(a["row"].to_numpy(), b["row"].to_numpy()):
        raise SystemExit(f"{run_a} and {run_b} were scored on different test rows")

    ca = ((a["prob"] >= 0.5).astype(int) == a["label"]).to_numpy()
    cb = ((b["prob"] >= 0.5).astype(int) == b["label"]).to_numpy()
    n01 = int((~ca & cb).sum())   # only b right
    n10 = int((ca & ~cb).sum())   # only a right
    if n01 + n10 == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(n10, n01 + n10, 0.5).pvalue)
    return {
        "run_a": run_a, "run_b": run_b,
        "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
        "only_a_right": n10, "only_b_right": n01,
        "p_value": p_value, "significant": bool(p_value < ALPHA),
    }


# --------------------------------------------------------------------------- #
# Across architectures
# --------------------------------------------------------------------------- #

def friedman_over_architectures(runs: pd.DataFrame, label_mode: str = "soft",
                                metric: str = "test_accuracy") -> dict:
    """Friedman test with seeds as blocks and architectures as treatments.

    This is the standard non-parametric route for comparing several classifiers
    over several trials (Demsar 2006). The post-hoc part is the Nemenyi critical
    difference, which is what the critical-difference diagram in the paper draws.
    """
    sub = runs[(runs["label_mode"] == label_mode) & (runs["policy"] == "d4")
               & (runs["finetune"] == "full") & (runs["train_size"] == 0)
               & (~runs["orientation_pooled"].astype(bool))]
    wide = sub.pivot_table(index="seed", columns="arch", values=metric)
    wide = wide.dropna(axis=1, how="any").dropna(axis=0, how="any")
    if wide.shape[1] < 3 or wide.shape[0] < 3:
        return {"error": "need at least three architectures and three seeds",
                "shape": list(wide.shape)}

    statistic, p_value = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    # ranks within each seed, 1 = best
    ranks = wide.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean(axis=0).sort_values()

    k, n = wide.shape[1], wide.shape[0]
    cd = _nemenyi_cd(k, n)
    return {
        "metric": metric,
        "label_mode": label_mode,
        "n_architectures": int(k),
        "n_seeds": int(n),
        "friedman_statistic": float(statistic),
        "friedman_p": float(p_value),
        "mean_ranks": {a: float(r) for a, r in mean_ranks.items()},
        "critical_difference": cd,
        "means": {a: float(wide[a].mean()) for a in wide.columns},
    }


def _nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference, CD = q_alpha * sqrt(k(k+1)/6n).

    The q values for alpha = 0.05 are the studentised range critical values divided
    by sqrt(2), as tabulated in Demsar (2006). Past k = 20 we fall back on the
    Bonferroni-style normal approximation, which is slightly conservative.
    """
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
           9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
           15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}
    q = q05.get(k) or float(stats.norm.ppf(1 - alpha / (k * (k - 1))))
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


PAIR_KEYS = ("arch", "label_mode", "policy", "size", "finetune", "loss",
             "train_size", "seed", "orientation_pooled", "pretrained")


def wilcoxon_ablation(runs: pd.DataFrame, column: str, level_a, level_b,
                      metric: str = "test_accuracy") -> dict | None:
    """Paired signed-rank test on one ablation knob.

    Runs are paired on every other setting, so each pair differs in exactly the one
    under test. Duplicates on the pairing key are averaged first; that only happens
    if a configuration was somehow run twice, but silently broadcasting mismatched
    lengths would be worse than averaging.
    """
    keys = [k for k in PAIR_KEYS if k != column and k in runs.columns]
    frame = runs.dropna(subset=[metric])
    a = frame[frame[column] == level_a].groupby(keys, dropna=False)[metric].mean()
    b = frame[frame[column] == level_b].groupby(keys, dropna=False)[metric].mean()
    shared = a.index.intersection(b.index)
    if len(shared) < 5:
        return None
    va, vb = a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
    try:
        statistic, p_value = stats.wilcoxon(va, vb)
    except ValueError:  # every difference is exactly zero
        statistic, p_value = 0.0, 1.0
    return {
        "knob": column, "level_a": str(level_a), "level_b": str(level_b),
        "metric": metric, "n_pairs": int(len(shared)),
        "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
        "mean_difference": float((va - vb).mean()),
        "statistic": float(statistic), "p_value": float(p_value),
        "significant": bool(p_value < ALPHA),
    }


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment, in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, p[idx] * (m - rank)))
        adjusted[idx] = running
    return adjusted


def architecture_pairwise(runs: pd.DataFrame, label_mode: str = "soft") -> pd.DataFrame:
    """Holm-corrected pairwise McNemar tests between architectures.

    A signed-rank test over five seeds cannot reach p < 0.05 no matter how large the
    difference is: with five pairs the smallest attainable two-sided p value is
    0.0625. The Nemenyi critical difference has the same problem from the other
    direction, being very wide with five blocks. Both are still worth reporting --
    Friedman for whether architecture matters at all, Nemenyi for the ranking -- but
    the pairwise claims are made here instead, where the unit of replication is the
    test galaxy rather than the seed.

    For each architecture the per-seed probabilities are averaged first, so every
    architecture is represented by the same kind of object (a five-member ensemble
    of itself) and the comparison is not at the mercy of one lucky initialisation.
    """
    sub = runs[(runs["label_mode"] == label_mode) & (runs["policy"] == "d4")
               & (runs["finetune"] == "full") & (runs["train_size"] == 0)
               & (~runs["orientation_pooled"].astype(bool))]
    if "pretrained" in sub:
        sub = sub[sub["pretrained"].fillna(True).astype(bool)]

    pooled: dict[str, pd.DataFrame] = {}
    for arch, group in sub.groupby("arch"):
        frames = []
        for run_id in group["run_id"]:
            path = config.RUNS / run_id / "predictions_test.csv"
            if path.exists():
                frames.append(pd.read_csv(path).sort_values("row").reset_index(drop=True))
        if not frames:
            continue
        base = frames[0][["row", "label"]].copy()
        base["prob"] = np.mean([f["prob"].to_numpy() for f in frames], axis=0)
        base["n_seeds"] = len(frames)
        pooled[arch] = base

    archs = sorted(pooled)
    if len(archs) < 2:
        return pd.DataFrame()

    rows = []
    for a, b in itertools.combinations(archs, 2):
        fa, fb = pooled[a], pooled[b]
        if not np.array_equal(fa["row"].to_numpy(), fb["row"].to_numpy()):
            continue
        ca = ((fa["prob"] >= 0.5).astype(int) == fa["label"]).to_numpy()
        cb = ((fb["prob"] >= 0.5).astype(int) == fb["label"]).to_numpy()
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p = 1.0 if n10 + n01 == 0 else float(stats.binomtest(n10, n10 + n01, 0.5).pvalue)
        rows.append({"arch_a": a, "arch_b": b,
                     "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
                     "only_a_right": n10, "only_b_right": n01, "p_raw": p,
                     "n_seeds": int(fa["n_seeds"].iloc[0])})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_raw"].to_numpy())
    out["significant"] = out["p_holm"] < ALPHA
    out["label_mode"] = label_mode
    return out.sort_values("p_holm").reset_index(drop=True)


# --------------------------------------------------------------------------- #

def paired_runs_for_mcnemar(runs: pd.DataFrame) -> list[tuple[str, str]]:
    """The comparisons the paper actually makes, on seed 0.

    Everything against everything would be hundreds of tests; we test the claims:
    hard vs soft labels per architecture, no augmentation vs d4, learnt invariance
    vs built-in invariance, and the best transformer against the best convnet.
    """
    pairs = []
    seed0 = runs[(runs["seed"] == 0) & (runs["train_size"] == 0)]

    def find(**kw):
        sub = seed0
        for k, v in kw.items():
            sub = sub[sub[k] == v]
        return sub["run_id"].tolist()

    for arch in sorted(runs["arch"].dropna().unique()):
        hard = find(arch=arch, label_mode="hard", policy="d4", finetune="full",
                    orientation_pooled=False)
        soft = find(arch=arch, label_mode="soft", policy="d4", finetune="full",
                    orientation_pooled=False)
        if hard and soft:
            pairs.append((soft[0], hard[0]))

        plain = find(arch=arch, label_mode="soft", policy="none", finetune="full",
                     orientation_pooled=False)
        d4 = find(arch=arch, label_mode="soft", policy="d4", finetune="full",
                  orientation_pooled=False)
        if plain and d4:
            pairs.append((d4[0], plain[0]))

        pooled = find(arch=arch, label_mode="soft", policy="none",
                      orientation_pooled=True)
        if pooled and d4:
            pairs.append((pooled[0], d4[0]))

    # best convnet vs best transformer under the reference protocol
    from src import registry

    ref = runs[(runs["label_mode"] == "soft") & (runs["policy"] == "d4")
               & (runs["train_size"] == 0) & (~runs["orientation_pooled"].astype(bool))]
    if not ref.empty:
        ref = ref.assign(family=ref["arch"].map(registry.family_of))
        best = {}
        for family, group in ref.groupby("family"):
            per_arch = group.groupby("arch")["test_accuracy"].mean().sort_values()
            if per_arch.empty:
                continue
            arch = per_arch.index[-1]
            rows = group[(group["arch"] == arch) & (group["seed"] == 0)]["run_id"].tolist()
            if rows:
                best[family] = rows[0]
        for fa, fb in itertools.combinations(sorted(best), 2):
            pairs.append((best[fa], best[fb]))

    # drop duplicates while keeping the order
    seen, out = set(), []
    for pair in pairs:
        key = tuple(sorted(pair))
        if key not in seen:
            seen.add(key)
            out.append(pair)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-bootstrap", action="store_true",
                    help="the bootstrap is the slow part; skip it when iterating")
    args = ap.parse_args()

    ensure_dir(config.RESULTS)
    runs_path = config.RESULTS / "runs.csv"
    if not runs_path.exists():
        raise SystemExit(f"missing {runs_path}; run src.analysis first")
    runs = pd.read_csv(runs_path)
    runs["orientation_pooled"] = runs["orientation_pooled"].fillna(False).astype(bool)

    if not args.skip_bootstrap:
        rows = []
        # one bootstrap per configuration is enough; use seed 0 as the representative
        wanted = runs[runs["seed"] == 0]["run_id"].tolist()
        print(f"bootstrapping {len(wanted)} runs with {args.n_boot} resamples")
        for i, run_id in enumerate(wanted, 1):
            row = bootstrap_metrics(run_id, args.n_boot)
            if row:
                rows.append(row)
            if i % 20 == 0:
                print(f"  {i} / {len(wanted)}", flush=True)
        pd.DataFrame(rows).to_csv(config.RESULTS / "bootstrap.csv", index=False)

    pairs = paired_runs_for_mcnemar(runs)
    print(f"running {len(pairs)} McNemar tests")
    rows = [r for r in (mcnemar(a, b) for a, b in pairs) if r]
    pd.DataFrame(rows).to_csv(config.RESULTS / "mcnemar.csv", index=False)

    friedman = {
        "soft_accuracy": friedman_over_architectures(runs, "soft", "test_accuracy"),
        "hard_accuracy": friedman_over_architectures(runs, "hard", "test_accuracy"),
        "soft_auc": friedman_over_architectures(runs, "soft", "test_auc_roc"),
    }
    write_json(config.RESULTS / "friedman.json", friedman)
    if "mean_ranks" in friedman["soft_accuracy"]:
        f = friedman["soft_accuracy"]
        print(f"Friedman over {f['n_architectures']} architectures, "
              f"{f['n_seeds']} seeds: p = {f['friedman_p']:.3g}, CD = "
              f"{f['critical_difference']:.2f}")

    pairwise = architecture_pairwise(runs, "soft")
    if not pairwise.empty:
        pairwise.to_csv(config.RESULTS / "architecture_pairwise.csv", index=False)
        print(f"{int(pairwise['significant'].sum())} of {len(pairwise)} architecture "
              f"pairs separate after Holm correction")

    tests = []
    for column, a, b in [
        ("label_mode", "soft", "hard"),
        ("label_mode", "soft_w", "soft"),
        ("label_mode", "hard_conf", "hard"),
        ("policy", "d4", "none"),
        ("policy", "d4", "flip"),
        ("policy", "d4_photo", "d4"),
        ("finetune", "full", "frozen"),
        ("finetune", "full", "partial"),
        ("loss", "focal", "bce"),
        ("loss", "smooth", "bce"),
        ("orientation_pooled", True, False),
        ("pretrained", True, False),
    ]:
        for metric in ("test_accuracy", "test_auc_roc", "test_ece"):
            result = wilcoxon_ablation(runs, column, a, b, metric)
            if result:
                tests.append(result)
    pd.DataFrame(tests).to_csv(config.RESULTS / "wilcoxon.csv", index=False)
    print(f"wrote {len(tests)} paired tests")


if __name__ == "__main__":
    main()
