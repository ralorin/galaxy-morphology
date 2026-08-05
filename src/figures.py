"""Every figure in the paper, drawn from the tables in $GZM_WORK/results.

    python -m src.figures --out /path/to/paper5
    python -m src.figures --out /path/to/paper5 --only agreement curve

Vector PDF, one file per figure, Type-1 fonts, no titles inside the axes (the
caption carries that). A figure whose inputs are missing is skipped with a note
rather than crashing the whole run, so this can be called while the sweep is still
going.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.common import ensure_dir, load_table, read_json, risk_coverage_curve
from src.registry import ARCHITECTURES, PRETTY_ARCH, PRETTY_FAMILY, REGISTRY, family_of
from src.tables import PRETTY_POLICY, reference

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

FAMILY_COLOUR = {"custom": "#8c8c8c", "cnn": "#1f6fb4", "transformer": "#c1512c"}
MODE_COLOUR = {"hard": "#4a4a4a", "soft": "#1f6fb4", "hard_conf": "#7fa8c9",
               "soft_w": "#c1512c"}


def _save(fig, out_dir: Path, name: str) -> None:
    path = ensure_dir(out_dir) / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def _runs() -> pd.DataFrame:
    runs = pd.read_csv(config.RESULTS / "runs.csv")
    runs["orientation_pooled"] = runs["orientation_pooled"].fillna(False).astype(bool)
    runs["train_size"] = runs["train_size"].fillna(0).astype(int)
    return runs


def _optional(name: str) -> pd.DataFrame | None:
    path = config.RESULTS / name
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------- #
# 1. The dataset and the disagreement it carries
# --------------------------------------------------------------------------- #

def fig_dataset(out_dir: Path) -> None:
    table = load_table("gz2")
    images = np.load(config.ARRAYS / "gz2_images.npy", mmap_mode="r")

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(table["p_featured"], bins=60, color="#1f6fb4", alpha=0.85)
    ax.set_xlabel("vote fraction $p$ (featured)")
    ax.set_ylabel("galaxies")
    ax.set_title("(a) volunteer votes", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(table["agreement"], bins=50, color="#c1512c", alpha=0.85)
    ax.axvline(0.6, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"agreement $|2p-1|$")
    ax.set_ylabel("galaxies")
    ax.set_title("(b) how much they agreed", loc="left")

    ax = fig.add_subplot(gs[0, 2])
    bins = np.array(config.AGREEMENT_BINS)
    idx = np.clip(np.digitize(table["agreement"], bins[1:-1]), 0, len(bins) - 2)
    share = [float((idx == b).mean()) for b in range(len(bins) - 1)]
    labels = [f"{bins[b]:.1f}-{bins[b + 1]:.1f}" for b in range(len(bins) - 1)]
    ax.bar(labels, share, color="#4a4a4a", alpha=0.85)
    ax.set_ylabel("share of the catalogue")
    ax.set_xlabel("agreement bin")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("(c) bins used throughout", loc="left")

    # a strip of galaxies from unanimous to contested
    rng = np.random.default_rng(config.SEED)
    picks = []
    for b in range(len(bins) - 1):
        sub = table[(table["agreement"] >= bins[b]) & (table["agreement"] < bins[b + 1] + 1e-9)]
        if len(sub):
            picks.append(sub.iloc[rng.integers(len(sub))])
    inner = gs[1, :].subgridspec(1, len(picks), wspace=0.08)
    for k, row in enumerate(picks):
        ax = fig.add_subplot(inner[0, k])
        ax.imshow(np.asarray(images[int(row["row"])]))
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"$p$={row['p_featured']:.2f}\n$N$={int(row['votes'])}", fontsize=7)
    _save(fig, out_dir, "fig_dataset")


# --------------------------------------------------------------------------- #
# 2. Where the error lives
# --------------------------------------------------------------------------- #

def fig_agreement(out_dir: Path) -> None:
    agr = _optional("agreement.csv")
    if agr is None:
        print("skipping fig_agreement: agreement.csv missing")
        return
    ceiling = read_json(config.RESULTS / "ceiling.json")["raw"] \
        if (config.RESULTS / "ceiling.json").exists() else None

    sub = agr[(agr["policy"] == "d4") & (agr["finetune"] == "full")
              & (agr["train_size"].fillna(0) == 0)]
    if sub.empty:
        print("skipping fig_agreement: no reference runs")
        return

    def centres_of(index):
        return [(config.AGREEMENT_BINS[int(b)] + config.AGREEMENT_BINS[int(b) + 1]) / 2
                for b in index]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 0.62], "hspace": 0.18,
                                          "wspace": 0.12})

    for col, mode in enumerate(("hard", "soft")):
        part = sub[sub["label_mode"] == mode]
        ax = axes[0, col]
        if part.empty:
            ax.axis("off"); axes[1, col].axis("off")
            continue

        for family in ("custom", "cnn", "transformer"):
            archs = [a for a in ARCHITECTURES if family_of(a) == family]
            block = part[part["arch"].isin(archs)]
            if block.empty:
                continue
            grouped = block.groupby("bin")["accuracy"]
            centres = centres_of(grouped.mean().index)
            ax.plot(centres, grouped.mean().to_numpy(), "-o", ms=3.5,
                    color=FAMILY_COLOUR[family], label=PRETTY_FAMILY[family])
            ax.fill_between(centres,
                            (grouped.mean() - grouped.std()).to_numpy(),
                            (grouped.mean() + grouped.std()).to_numpy(),
                            color=FAMILY_COLOUR[family], alpha=0.15, lw=0)

        # The class prior swings between bins, so accuracy on its own is not
        # comparable across them. Draw the within-bin majority baseline: the gap
        # between a curve and this line is what the model actually contributes.
        if "majority_baseline" in part:
            base = part.groupby("bin")["majority_baseline"].mean()
            ax.plot(centres_of(base.index), base.to_numpy(), "--", color="#999999",
                    lw=1.0, label="majority baseline in bin")

        if ceiling:
            ax.axhline(ceiling["bayes_accuracy"], color="k", ls=":", lw=0.9)
            ax.text(0.02, ceiling["bayes_accuracy"] + 0.008,
                    r"$A^\star$ (whole test set)", fontsize=6.5)

        ax.set_title(f"({'a' if mode == 'hard' else 'b'}) {mode} labels", loc="left")
        ax.set_ylim(0.35, 1.02)
        if col == 0:
            ax.set_ylabel("test accuracy")
        else:
            ax.tick_params(labelleft=False)

        # lower panel: how the test set and the errors are distributed over the bins
        ax2 = axes[1, col]
        share = part.groupby("bin")["share"].mean()
        x = np.array(centres_of(share.index))
        ax2.bar(x - 0.04, share.to_numpy(), width=0.075, color="#bbbbbb",
                label="share of test set")
        if "error_share" in part:
            err = part.groupby("bin")["error_share"].mean()
            ax2.bar(x + 0.04, err.to_numpy(), width=0.075, color="#c1512c",
                    label="share of all errors")
        ax2.set_ylim(0, max(0.6, float(share.max()) * 1.15))
        ax2.set_xlabel(r"volunteer agreement $|2p-1|$")
        if col == 0:
            ax2.set_ylabel("fraction")
            ax2.legend(frameon=False, fontsize=7)
        else:
            ax2.tick_params(labelleft=False)

    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=7)
    _save(fig, out_dir, "fig_agreement")


# --------------------------------------------------------------------------- #
# 3. Learning curves: is it data or labels?
# --------------------------------------------------------------------------- #

def fig_curve(out_dir: Path) -> None:
    runs = _runs()
    sub = runs[(runs["policy"] == "d4") & (runs["finetune"] == "full")
               & (runs["loss"] == "bce") & (~runs["orientation_pooled"])]
    archs = [a for a in ("resnet50", "vit_small") if a in set(sub["arch"])]
    if not archs:
        print("skipping fig_curve: no runs")
        return

    fig, axes = plt.subplots(1, len(archs), figsize=(3.6 * len(archs), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    ceiling = read_json(config.RESULTS / "ceiling.json")["raw"] \
        if (config.RESULTS / "ceiling.json").exists() else None

    full = sub["train_size"].replace(0, np.nan).max()
    for ax, arch in zip(axes, archs):
        for mode in ("hard", "soft"):
            part = sub[(sub["arch"] == arch) & (sub["label_mode"] == mode)]
            if part.empty:
                continue
            part = part.assign(n=part["train_size"].replace(0, np.nan).fillna(
                _train_split_size()))
            g = part.groupby("n")["test_accuracy"]
            ax.errorbar(g.mean().index, g.mean().to_numpy(),
                        yerr=g.std().fillna(0).to_numpy(), fmt="-o", ms=3.5, lw=1.2,
                        capsize=2, color=MODE_COLOUR[mode], label=f"{mode} labels")
        if ceiling:
            ax.axhline(ceiling["bayes_accuracy"], color="k", ls=":", lw=0.9,
                       label="vote-model ceiling")
        ax.set_xscale("log")
        ax.set_xlabel("training galaxies")
        ax.set_title(PRETTY_ARCH.get(arch, arch), loc="left")
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(loc="lower right", frameon=False)
    _save(fig, out_dir, "fig_curve")


def _train_split_size() -> int:
    meta = config.ARRAYS / "gz2_meta.json"
    if meta.exists():
        return int(read_json(meta).get("split_counts", {}).get("train", 100000))
    return 100000


# --------------------------------------------------------------------------- #
# 4. Calibration
# --------------------------------------------------------------------------- #

def fig_calibration(out_dir: Path) -> None:
    cal = _optional("calibration.csv")
    runs = _runs()
    if cal is None or cal.empty:
        print("skipping fig_calibration: calibration.csv missing")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k:", lw=0.9)
    for mode in ("hard", "soft"):
        part = cal[cal["label_mode"] == mode]
        if part.empty:
            continue
        g = part.groupby("confidence")["observed"].mean()
        ax.plot(g.index, g.to_numpy(), "-o", ms=3.5, color=MODE_COLOUR[mode],
                label=f"{mode} labels")
    ax.set_xlabel("predicted probability of featured")
    ax.set_ylabel("observed frequency")
    ax.set_title("(a) reliability", loc="left")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ref = reference(runs)
    archs = [a for a in ARCHITECTURES if a in set(ref["arch"])]
    width = 0.38
    x = np.arange(len(archs))
    for k, mode in enumerate(("hard", "soft")):
        vals, errs = [], []
        for arch in archs:
            rows = ref[(ref["arch"] == arch) & (ref["label_mode"] == mode)]
            vals.append(rows["test_ece"].mean() if not rows.empty else np.nan)
            errs.append(rows["test_ece"].std(ddof=1) if len(rows) > 1 else 0.0)
        ax.bar(x + (k - 0.5) * width, vals, width, yerr=errs, capsize=1.5,
               color=MODE_COLOUR[mode], label=f"{mode} labels")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_ARCH.get(a, a) for a in archs], rotation=60,
                       ha="right", fontsize=7)
    ax.set_ylabel("expected calibration error")
    ax.set_title("(b) ECE by architecture", loc="left")
    ax.legend(frameon=False)
    _save(fig, out_dir, "fig_calibration")


# --------------------------------------------------------------------------- #
# 5. Selective prediction
# --------------------------------------------------------------------------- #

def fig_selective(out_dir: Path) -> None:
    runs = _runs()
    ref = reference(runs)
    picks = []
    for family in ("cnn", "transformer"):
        archs = [a for a in ARCHITECTURES if family_of(a) == family]
        block = ref[(ref["arch"].isin(archs)) & (ref["label_mode"] == "soft")
                    & (ref["seed"] == 0)]
        if block.empty:
            continue
        best = block.loc[block["test_accuracy"].idxmax()]
        picks.append((best["arch"], "soft", best["run_id"]))
        hard = ref[(ref["arch"] == best["arch"]) & (ref["label_mode"] == "hard")
                   & (ref["seed"] == 0)]
        if not hard.empty:
            picks.append((best["arch"], "hard", hard["run_id"].iloc[0]))
    if not picks:
        print("skipping fig_selective: no runs")
        return

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for arch, mode, run_id in picks:
        path = config.RUNS / run_id / "predictions_test.csv"
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        cov, acc, _ = risk_coverage_curve(pred["label"], pred["prob"])
        step = max(1, len(cov) // 2000)
        ax.plot(cov[::step], acc[::step], lw=1.3,
                ls="-" if mode == "soft" else "--",
                color=FAMILY_COLOUR[family_of(arch)],
                label=f"{PRETTY_ARCH.get(arch, arch)}, {mode}")
    for target in (0.99, 0.95):
        ax.axhline(target, color="k", ls=":", lw=0.7)
        ax.text(0.02, target + 0.002, f"{target:.2f}", fontsize=6.5)
    ax.set_xlabel("coverage (fraction classified automatically)")
    ax.set_ylabel("accuracy on the covered part")
    ax.set_xlim(0.05, 1.0)
    ax.legend(frameon=False, loc="lower left")
    _save(fig, out_dir, "fig_selective")


# --------------------------------------------------------------------------- #
# 6. Orientation
# --------------------------------------------------------------------------- #

def fig_orientation(out_dir: Path) -> None:
    runs = _runs()
    sub = runs[(runs["label_mode"] == "soft") & (runs["finetune"] == "full")
               & (runs["loss"] == "bce") & (runs["train_size"] == 0)]
    archs = [a for a in ("resnet50", "convnext_tiny", "vit_small") if a in set(sub["arch"])]
    if not archs:
        print("skipping fig_orientation: no runs")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    policies = ["none", "flip", "d4", "d4_photo"]
    x = np.arange(len(policies))
    width = 0.8 / max(1, len(archs))
    for k, arch in enumerate(archs):
        vals, errs = [], []
        for policy in policies:
            rows = sub[(sub["arch"] == arch) & (sub["policy"] == policy)
                       & (~sub["orientation_pooled"])]
            vals.append(rows["test_accuracy"].mean() if not rows.empty else np.nan)
            errs.append(rows["test_accuracy"].std(ddof=1) if len(rows) > 1 else 0.0)
        axes[0].bar(x + (k - (len(archs) - 1) / 2) * width, vals, width, yerr=errs,
                    capsize=1.5, label=PRETTY_ARCH.get(arch, arch),
                    color=FAMILY_COLOUR[family_of(arch)],
                    alpha=0.55 + 0.45 * k / max(1, len(archs) - 1))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([PRETTY_POLICY[p] for p in policies], rotation=20)
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("(a) augmentation policy", loc="left")
    axes[0].legend(frameon=False, fontsize=7)
    lo = np.nanmin([sub["test_accuracy"].min(), 1.0])
    axes[0].set_ylim(max(0.5, lo - 0.03), sub["test_accuracy"].max() + 0.01)

    ax = axes[1]
    labels, values = [], []
    for arch in archs:
        for policy, pooled in (("none", False), ("d4", False), ("none", True)):
            rows = sub[(sub["arch"] == arch) & (sub["policy"] == policy)
                       & (sub["orientation_pooled"] == pooled)]
            if rows.empty:
                continue
            tag = "built-in" if pooled else PRETTY_POLICY[policy]
            labels.append(f"{PRETTY_ARCH.get(arch, arch)}\n{tag}")
            values.append(rows["d4_invariance_error"].mean())
    ax.bar(range(len(values)), values, color="#4a4a4a", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel(r"$\varepsilon_{D_4}$")
    ax.set_title("(b) how far from invariant", loc="left")
    _save(fig, out_dir, "fig_orientation")


# --------------------------------------------------------------------------- #
# 7. Cost
# --------------------------------------------------------------------------- #

def fig_pareto(out_dir: Path) -> None:
    runs = _runs()
    ref = reference(runs)
    ref = ref[ref["label_mode"] == "soft"]
    if ref.empty:
        print("skipping fig_pareto: no runs")
        return
    g = ref.groupby("arch").agg(
        accuracy=("test_accuracy", "mean"),
        throughput=("images_per_second", "mean"),
        params=("params", "first"),
    ).dropna()

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for arch, row in g.iterrows():
        family = family_of(arch)
        ax.scatter(row["throughput"], row["accuracy"],
                   s=18 + 90 * np.sqrt(row["params"] / g["params"].max()),
                   color=FAMILY_COLOUR[family], alpha=0.8, edgecolor="w", lw=0.6,
                   zorder=3)
        ax.annotate(PRETTY_ARCH.get(arch, arch), (row["throughput"], row["accuracy"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=6.5)

    # the Pareto front: nothing to the upper right of these
    front = g.sort_values("throughput", ascending=False)
    best, xs, ys = -np.inf, [], []
    for _, row in front.iterrows():
        if row["accuracy"] > best:
            best = row["accuracy"]
            xs.append(row["throughput"]); ys.append(row["accuracy"])
    ax.step(xs, ys, where="post", color="k", lw=0.8, ls="--", zorder=2,
            label="Pareto front")
    ax.set_xscale("log")
    ax.set_xlabel("inference throughput (images/s, one H100)")
    ax.set_ylabel("test accuracy")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=PRETTY_FAMILY[f])
               for f, c in FAMILY_COLOUR.items()]
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=7)
    _save(fig, out_dir, "fig_pareto")


# --------------------------------------------------------------------------- #
# 8. Cross-survey
# --------------------------------------------------------------------------- #

def fig_cross_survey(out_dir: Path) -> None:
    cross = _optional("cross_survey.csv")
    if cross is None or cross.empty:
        print("skipping fig_cross_survey: cross_survey.csv missing")
        return
    cross["orientation_pooled"] = cross["orientation_pooled"].fillna(False).astype(bool)
    sub = cross[~cross["orientation_pooled"]]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for mode, marker in (("hard", "s"), ("soft", "o")):
        part = sub[sub["label_mode"] == mode]
        if part.empty:
            continue
        g = part.groupby("arch")[["test_accuracy", "decals_accuracy"]].mean()
        ax.scatter(g["test_accuracy"], g["decals_accuracy"], marker=marker, s=30,
                   color=MODE_COLOUR[mode], alpha=0.85, label=f"{mode} labels",
                   zorder=3)
        for arch, row in g.iterrows():
            ax.annotate(PRETTY_ARCH.get(arch, arch),
                        (row["test_accuracy"], row["decals_accuracy"]),
                        textcoords="offset points", xytext=(5, -1), fontsize=6)
    lims = [min(sub["decals_accuracy"].min(), sub["test_accuracy"].min()) - 0.02,
            max(sub["test_accuracy"].max(), sub["decals_accuracy"].max()) + 0.02]
    ax.plot(lims, lims, "k:", lw=0.9, label="no loss in transfer")
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel("accuracy on SDSS (Galaxy Zoo 2)")
    ax.set_ylabel("accuracy on DECaLS (Galaxy10)")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    _save(fig, out_dir, "fig_cross_survey")


# --------------------------------------------------------------------------- #
# 9. Faithfulness of the explanations
# --------------------------------------------------------------------------- #

def fig_faithfulness(out_dir: Path) -> None:
    xai = _optional("xai_summary.csv")
    if xai is None or xai.empty:
        print("skipping fig_faithfulness: xai_summary.csv missing")
        return
    runs = _runs()
    merged = xai.merge(runs[["run_id", "test_accuracy"]], on="run_id", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axes[0]
    for family, colour in FAMILY_COLOUR.items():
        part = merged[merged["arch"].map(family_of) == family]
        if part.empty:
            continue
        ax.scatter(part["test_accuracy"], part["background_reliance"], s=26,
                   color=colour, alpha=0.85, label=PRETTY_FAMILY[family], zorder=3)
        for _, row in part.iterrows():
            ax.annotate(PRETTY_ARCH.get(row["arch"], row["arch"]),
                        (row["test_accuracy"], row["background_reliance"]),
                        textcoords="offset points", xytext=(5, -1), fontsize=6)
    ax.set_xlabel("test accuracy")
    ax.set_ylabel("attribution mass outside the galaxy")
    ax.set_title("(a) accuracy is not groundedness", loc="left")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    x = np.arange(len(merged))
    order = merged.sort_values("background_reliance")
    ax.barh(x, order["background_reliance_high_agreement"], height=0.4,
            color="#1f6fb4", label="volunteers agreed")
    ax.barh(x + 0.42, order["background_reliance_low_agreement"], height=0.4,
            color="#c1512c", label="volunteers disagreed")
    ax.set_yticks(x + 0.21)
    ax.set_yticklabels([f"{PRETTY_ARCH.get(a, a)} ({m})"
                        for a, m in zip(order["arch"], order["label_mode"])],
                       fontsize=6.5)
    ax.set_xlabel("attribution mass outside the galaxy")
    ax.set_title("(b) split by agreement", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _save(fig, out_dir, "fig_faithfulness")


# --------------------------------------------------------------------------- #
# 10. Critical difference diagram
# --------------------------------------------------------------------------- #

def fig_critical_difference(out_dir: Path) -> None:
    path = config.RESULTS / "friedman.json"
    if not path.exists():
        print("skipping fig_critical_difference: friedman.json missing")
        return
    result = read_json(path).get("soft_accuracy", {})
    if "mean_ranks" not in result:
        print("skipping fig_critical_difference: not enough runs")
        return

    ranks = pd.Series(result["mean_ranks"]).sort_values()
    cd = result["critical_difference"]

    fig, ax = plt.subplots(figsize=(6.4, 2.2 + 0.16 * len(ranks)))
    ax.set_xlim(0.5, len(ranks) + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.78, 1, len(ranks), color="k", lw=1.0)
    for tick in range(1, len(ranks) + 1):
        ax.vlines(tick, 0.76, 0.80, color="k", lw=1.0)
        ax.text(tick, 0.83, str(tick), ha="center", fontsize=7)
    ax.text(len(ranks) / 2 + 0.5, 0.92, "mean rank (1 = best)", ha="center", fontsize=8)

    # the CD bar
    ax.hlines(0.68, 1, 1 + cd, color="k", lw=1.6)
    ax.vlines([1, 1 + cd], 0.66, 0.70, color="k", lw=1.6)
    ax.text(1 + cd / 2, 0.60, f"CD = {cd:.2f}", ha="center", fontsize=7)

    step = 0.5 / max(1, len(ranks))
    for i, (arch, rank) in enumerate(ranks.items()):
        y = 0.50 - i * step
        side = -1 if i < len(ranks) / 2 else 1
        anchor = 0.8 if side < 0 else len(ranks) + 0.2
        ax.plot([rank, rank, anchor], [0.76, y, y], color="#666666", lw=0.8)
        ax.text(anchor + 0.12 * side, y,
                f"{PRETTY_ARCH.get(arch, arch)} ({rank:.2f})",
                va="center", ha="left" if side > 0 else "right", fontsize=7)
    _save(fig, out_dir, "fig_critical_difference")


# --------------------------------------------------------------------------- #
# 11. The protocol, drawn
# --------------------------------------------------------------------------- #

def fig_pipeline(out_dir: Path) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.4)
    ax.axis("off")

    boxes = [
        (0.1, 1.15, 2.0, "Galaxy Zoo 2\n(SDSS cutouts +\nvolunteer votes)", "#dce7f2"),
        (2.4, 1.15, 1.9, "task-01 vote\nfraction $p$,\nvote count $N$", "#dce7f2"),
        (4.6, 2.05, 2.2, "training target:\nhard $1[p>0.5]$\nor soft $p$", "#f2e6dc"),
        (4.6, 0.25, 2.2, "vote model:\n$\\pi(p,N)$ and the\nachievable ceiling", "#eaeaea"),
        (7.1, 1.15, 2.1, "12 backbones\n$D_4$ augmentation\nor built-in\ninvariance", "#f2e6dc"),
        (9.5, 1.15, 2.3, "agreement-resolved\nerror, calibration,\nselective prediction,\nexplanations", "#dcf2e2"),
    ]
    for x, y, w, text, colour in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, 1.15, boxstyle="round,pad=0.06",
                                    linewidth=0.7, edgecolor="#555555",
                                    facecolor=colour))
        ax.text(x + w / 2, y + 0.575, text, ha="center", va="center", fontsize=6.8)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.8, color="#555555"))

    arrow(2.1, 1.73, 2.4, 1.73)
    arrow(4.3, 1.73, 4.6, 2.63)
    arrow(4.3, 1.73, 4.6, 0.83)
    arrow(6.8, 2.63, 7.1, 1.9)
    arrow(9.2, 1.73, 9.5, 1.73)
    arrow(6.8, 0.83, 9.5, 1.4)
    ax.text(11.0, 0.15, "held out: Galaxy10 DECaLS", fontsize=6.5, ha="right",
            style="italic")
    _save(fig, out_dir, "fig_pipeline")


# --------------------------------------------------------------------------- #

FIGURES = {
    "dataset": fig_dataset,
    "agreement": fig_agreement,
    "curve": fig_curve,
    "calibration": fig_calibration,
    "selective": fig_selective,
    "orientation": fig_orientation,
    "pareto": fig_pareto,
    "cross_survey": fig_cross_survey,
    "faithfulness": fig_faithfulness,
    "critical_difference": fig_critical_difference,
    "pipeline": fig_pipeline,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=config.WORK / "paper_assets")
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out / "figures")
    for name in (args.only or FIGURES):
        try:
            FIGURES[name](out_dir)
        except (FileNotFoundError, KeyError, SystemExit) as exc:
            print(f"skipping fig_{name}: {exc}")


if __name__ == "__main__":
    main()
