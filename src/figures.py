"""Every figure in the paper, drawn from the tables in $GZM_WORK/results.

    python -m src.figures --out /path/to/paper5
    python -m src.figures --out /path/to/paper5 --only agreement curve

Vector PDF, one file per figure, Type-1 fonts, no titles inside the axes (the
caption carries that). A figure whose inputs are missing is skipped with a note
rather than crashing the whole run, so this can be called while the sweep is still
going.

On the visual conventions
-------------------------
Three categorical colours, assigned to entities in a fixed order and never cycled:
they carry either the architecture family or the training target, and which one is
always named in the legend. Three is not an accident. It is the largest set that
clears the colour-vision-deficiency and normal-vision separation floors on a white
surface when every pair can appear together, which is the case in the scatter
plots; a fourth hue would fail. Wherever a figure needs more distinctions than
that, it gets more panels instead of more colours.

The rest of the ink is deliberately quiet: hairline solid gridlines one step off
the surface, axis text in a muted grey, and no colour on any text at all. A series
is identified by a coloured mark beside a label, never by colouring the label,
because the lightest of the three hues is not legible as text. That hue also sits
below 3:1 against white, so wherever it appears it carries a direct label as well
as a legend entry, and every value in every figure also exists in one of the
tables.

There are no dual-axis plots here. Where two quantities on different scales matter
-- accuracy and the share of the test set, in the agreement figure -- they get
stacked panels sharing an x-axis rather than two y-scales on one frame.
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

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

# Categorical slots, in fixed order. Validated as a set on a white surface with
# every pair in play: worst CVD separation 9.2, worst normal-vision separation
# 24.0, both clear of their floors.
SLOT = ("#2a78d6", "#eb6834", "#1baf7a")   # blue, orange, aqua

# Chart chrome. Nothing here competes with the data.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECOND = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Entities keep their colour across every figure in which they appear.
FAMILY_COLOUR = {"custom": SLOT[0], "cnn": SLOT[1], "transformer": SLOT[2]}
MODE_COLOUR = {"hard": SLOT[0], "soft": SLOT[1],
               "hard_conf": SLOT[2], "soft_w": SLOT[2]}
PROBE_COLOUR = {"resnet50": SLOT[0], "convnext_tiny": SLOT[1], "vit_small": SLOT[2]}

# The one hue whose contrast against white is below 3:1; anything drawn in it also
# gets a direct label, which is the documented relief.
NEEDS_LABEL = SLOT[2]

# Two steps of the blue ramp, for the one contrast in this paper that is ordered
# rather than categorical: galaxies the volunteers agreed on against galaxies they
# did not. Using a second pair of categorical hues there would make blue mean an
# architecture family in one panel and an agreement level in the next; a one-hue
# ramp says "two levels of the same thing", which is what it is. The light step is
# the ordinal floor for a white surface.
ORDINAL = ("#86b6ef", "#1c5cab")

LINE_W = 1.5          # 2 px
MARKER = 5.0          # >= 8 px including its ring
RING_W = 1.2          # 2 px of surface around a marker
FILL_ALPHA = 0.10     # area washes, never a saturated block
BAR_MAX = 0.24        # bars stay thin; the leftover band is air

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linestyle": "-",          # solid hairline; dashed grids compete with data
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
})


def _line(ax, x, y, colour, label=None, band=None, marker="o", ls="-"):
    """One series: a 2 px line, ringed markers and an optional 10% wash."""
    if band is not None:
        lo, hi = band
        ax.fill_between(x, lo, hi, color=colour, alpha=FILL_ALPHA, lw=0)
    ax.plot(x, y, ls=ls, lw=LINE_W, color=colour, label=label,
            marker=marker, ms=MARKER, markerfacecolor=colour,
            markeredgecolor=SURFACE, markeredgewidth=RING_W,
            solid_capstyle="round", solid_joinstyle="round")


def _bars(ax, centres, values, colour, width, errs=None, label=None):
    """Thin bars with the band's leftover left as air, and no separating stroke."""
    ax.bar(centres, values, width=width, color=colour, label=label,
           yerr=errs, capsize=0, error_kw=dict(lw=0.8, ecolor=INK_MUTED))


def _tidy(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", color=INK_SECOND)
    ax.tick_params(length=0)


def _save(fig, out_dir: Path, name: str) -> None:
    path = ensure_dir(out_dir) / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def _runs() -> pd.DataFrame:
    runs = pd.read_csv(config.RESULTS / "runs.csv")
    runs["orientation_pooled"] = runs["orientation_pooled"].fillna(False).astype(bool)
    runs["train_size"] = runs["train_size"].fillna(0).astype(int)
    if "pretrained" not in runs:
        runs["pretrained"] = True
    runs["pretrained"] = runs["pretrained"].fillna(True).astype(bool)
    return runs


def _optional(name: str) -> pd.DataFrame | None:
    path = config.RESULTS / name
    return pd.read_csv(path) if path.exists() else None


def _bin_centres(index):
    return np.array([(config.AGREEMENT_BINS[int(b)] + config.AGREEMENT_BINS[int(b) + 1]) / 2
                     for b in index])


# --------------------------------------------------------------------------- #
# 1. The dataset and the disagreement it carries
# --------------------------------------------------------------------------- #

def fig_dataset(out_dir: Path) -> None:
    table = load_table("gz2")
    images = np.load(config.ARRAYS / "gz2_images.npy", mmap_mode="r")

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.5, wspace=0.32)

    # one series per panel, so no legend: the axis label says what is plotted
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(table["p_featured"], bins=60, color=SLOT[0])
    _tidy(ax, r"vote fraction $p$ (featured)", "galaxies", "(a) volunteer votes")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(table["agreement"], bins=50, color=SLOT[0])
    ax.axvline(0.6, color=INK_MUTED, lw=0.8)
    ax.annotate("confident\nthreshold", xy=(0.6, ax.get_ylim()[1] * 0.82),
                xytext=(4, 0), textcoords="offset points", fontsize=6.5,
                color=INK_MUTED, va="top")
    _tidy(ax, r"agreement $|2p-1|$", "galaxies", "(b) how much they agreed")

    ax = fig.add_subplot(gs[0, 2])
    edges = np.array(config.AGREEMENT_BINS)
    idx = np.clip(np.digitize(table["agreement"], edges[1:-1]), 0, len(edges) - 2)
    share = np.array([float((idx == b).mean()) for b in range(len(edges) - 1)])
    labels = [f"{edges[b]:.1f}" for b in range(len(edges) - 1)]
    ax.bar(np.arange(len(share)), share, width=0.55, color=SLOT[0])
    for i, v in enumerate(share):          # values on the caps, no y-axis needed
        ax.annotate(f"{100 * v:.0f}%", (i, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6.5, color=INK_SECOND)
    ax.set_xticks(np.arange(len(share)))
    ax.set_xticklabels(labels)
    ax.set_yticks([])
    ax.grid(False)
    _tidy(ax, "agreement bin (lower edge)", None, "(c) bins used throughout")

    rng = np.random.default_rng(config.SEED)
    picks = []
    for b in range(len(edges) - 1):
        sub = table[(table["agreement"] >= edges[b])
                    & (table["agreement"] < edges[b + 1] + 1e-9)]
        if len(sub):
            picks.append(sub.iloc[rng.integers(len(sub))])
    inner = gs[1, :].subgridspec(1, len(picks), wspace=0.08)
    for k, row in enumerate(picks):
        ax = fig.add_subplot(inner[0, k])
        ax.imshow(np.asarray(images[int(row["row"])]))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"$p$={row['p_featured']:.2f}   $N$={int(row['votes'])}",
                     fontsize=6.5, color=INK_SECOND)
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

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 0.6], "hspace": 0.15,
                                          "wspace": 0.10})

    for col, mode in enumerate(("hard", "soft")):
        part = sub[sub["label_mode"] == mode]
        ax = axes[0, col]
        if part.empty:
            ax.axis("off"); axes[1, col].axis("off")
            continue

        for family in ("custom", "cnn", "transformer"):
            block = part[part["arch"].isin([a for a in ARCHITECTURES
                                            if family_of(a) == family])]
            if block.empty:
                continue
            g = block.groupby("bin")["accuracy"]
            x = _bin_centres(g.mean().index)
            colour = FAMILY_COLOUR[family]
            y = g.mean().to_numpy()
            _line(ax, x, y, colour, label=PRETTY_FAMILY[family],
                  band=((g.mean() - g.std()).to_numpy(),
                        (g.mean() + g.std()).to_numpy()))
            # The aqua slot sits below 3:1 on white, so it also carries a direct
            # label. Put it at the left end, where the three curves are still far
            # apart; at the right end they converge and any label there detaches
            # from its line.
            if colour == NEEDS_LABEL and col == 1:
                ax.annotate(PRETTY_FAMILY[family], (x[0], y[0]),
                            textcoords="offset points", xytext=(6, 6),
                            ha="left", fontsize=6.5, color=INK_SECOND)

        if "majority_baseline" in part:
            base = part.groupby("bin")["majority_baseline"].mean()
            ax.plot(_bin_centres(base.index), base.to_numpy(), lw=1.0,
                    color=INK_MUTED, label="majority baseline in bin")

        if ceiling:
            ax.axhline(ceiling["bayes_accuracy"], lw=0.8, color=INK_MUTED, ls=(0, (4, 3)))
            ax.annotate(r"$A^\star$", (0.02, ceiling["bayes_accuracy"]),
                        textcoords="offset points", xytext=(0, 3), fontsize=7,
                        color=INK_SECOND)

        _tidy(ax, None, "test accuracy" if col == 0 else None,
              f"({'a' if col == 0 else 'b'}) {mode} labels")
        ax.set_ylim(0.35, 1.03)
        if col == 1:
            ax.tick_params(labelleft=False)

        ax2 = axes[1, col]
        share = part.groupby("bin")["share"].mean()
        x = _bin_centres(share.index)
        gap = 0.012                      # surface gap, not a stroke
        w = BAR_MAX / 2
        ax2.bar(x - w / 2 - gap / 2, share.to_numpy(), width=w, color=ORDINAL[0],
                label="share of test set")
        if "error_share" in part:
            err = part.groupby("bin")["error_share"].mean()
            ax2.bar(x + w / 2 + gap / 2, err.to_numpy(), width=w, color=ORDINAL[1],
                    label="share of all errors")
            top = err.idxmax()
            ax2.annotate(f"{100 * err.loc[top]:.0f}%",
                         (_bin_centres([top])[0] + w / 2 + gap / 2, err.loc[top]),
                         textcoords="offset points", xytext=(0, 3), ha="center",
                         fontsize=7, color=INK_SECOND)
        ax2.set_ylim(0, 1.0)
        _tidy(ax2, r"volunteer agreement $|2p-1|$",
              "fraction" if col == 0 else None)
        if col == 1:
            ax2.tick_params(labelleft=False)
        else:
            ax2.legend(loc="upper right", fontsize=7)

    axes[0, 0].legend(loc="lower right", fontsize=7)
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

    fig, axes = plt.subplots(1, len(archs), figsize=(3.7 * len(archs), 3.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    ceiling = read_json(config.RESULTS / "ceiling.json")["raw"] \
        if (config.RESULTS / "ceiling.json").exists() else None

    for k, (ax, arch) in enumerate(zip(axes, archs)):
        for mode in ("hard", "soft"):
            part = sub[(sub["arch"] == arch) & (sub["label_mode"] == mode)]
            if part.empty:
                continue
            part = part.assign(n=part["train_size"].replace(0, np.nan)
                               .fillna(_train_split_size()))
            g = part.groupby("n")["test_accuracy"]
            _line(ax, g.mean().index.to_numpy(), g.mean().to_numpy(),
                  MODE_COLOUR[mode], label=f"{mode} labels" if k == 0 else None,
                  band=((g.mean() - g.std().fillna(0)).to_numpy(),
                        (g.mean() + g.std().fillna(0)).to_numpy()))
        if ceiling:
            ax.axhline(ceiling["bayes_accuracy"], lw=0.8, color=INK_MUTED,
                       ls=(0, (4, 3)),
                       label=r"vote-model ceiling $A^\star$" if k == 0 else None)
        ax.set_xscale("log")
        _tidy(ax, "training galaxies",
              "test accuracy" if k == 0 else None, PRETTY_ARCH.get(arch, arch))
    axes[0].legend(loc="lower right", fontsize=7)
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
    ax.plot([0, 1], [0, 1], lw=0.8, color=INK_MUTED, ls=(0, (4, 3)),
            label="perfect calibration")
    # pool on the bin index, not on the mean confidence inside the bin: the latter
    # shifts slightly from run to run and grouping on it gives a ragged curve
    key = "bin" if "bin" in cal else "confidence"
    for mode in ("hard", "soft"):
        part = cal[cal["label_mode"] == mode]
        if part.empty:
            continue
        g = part.groupby(key).agg(x=("bin_centre", "mean") if "bin_centre" in part
                                  else ("confidence", "mean"),
                                  y=("observed", "mean")).dropna()
        _line(ax, g["x"].to_numpy(), g["y"].to_numpy(), MODE_COLOUR[mode],
              label=f"{mode} labels")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _tidy(ax, "predicted probability of featured", "observed frequency",
          "(a) reliability")
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[1]
    ref = reference(runs)
    archs = [a for a in ARCHITECTURES if a in set(ref["arch"])]
    x = np.arange(len(archs))
    gap, w = 0.04, 0.34
    for k, (mode, column, label) in enumerate([
            ("hard", "test_ece", "hard, as measured"),
            ("soft", "test_ece", "soft, as measured")]):
        vals = [ref[(ref["arch"] == a) & (ref["label_mode"] == mode)][column].mean()
                for a in archs]
        errs = [ref[(ref["arch"] == a) & (ref["label_mode"] == mode)][column].std(ddof=1)
                for a in archs]
        _bars(ax, x + (k - 0.5) * (w + gap), vals, MODE_COLOUR[mode], w,
              errs=errs, label=label)
    if "test_ece_calibrated" in ref:
        vals = [ref[ref["arch"] == a]["test_ece_calibrated"].mean() for a in archs]
        ax.plot(x, vals, ls="none", marker="_", ms=12, color=INK,
                markeredgewidth=1.4, label="either, after temperature scaling")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_ARCH.get(a, a) for a in archs], rotation=60,
                       ha="right", fontsize=6.5)
    _tidy(ax, None, "expected calibration error", "(b) by architecture")
    ax.legend(loc="upper left", fontsize=7)
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

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for arch, mode, run_id in picks:
        path = config.RUNS / run_id / "predictions_test.csv"
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        cov, acc, _ = risk_coverage_curve(pred["label"], pred["prob"])
        step = max(1, len(cov) // 2000)
        ax.plot(cov[::step], acc[::step], lw=LINE_W,
                ls="-" if mode == "soft" else (0, (4, 2)),
                color=FAMILY_COLOUR[family_of(arch)],
                label=f"{PRETTY_ARCH.get(arch, arch)}, {mode}",
                solid_capstyle="round")
    for target in (0.99, 0.95):
        ax.axhline(target, lw=0.7, color=INK_MUTED, ls=(0, (2, 3)))
        ax.annotate(f"{target:.2f}", (0.055, target), textcoords="offset points",
                    xytext=(0, 3), fontsize=6.5, color=INK_MUTED)
    _tidy(ax, "coverage (fraction classified automatically)",
          "accuracy on the covered part")
    ax.set_xlim(0.05, 1.0)
    ax.legend(loc="lower left", fontsize=7)
    _save(fig, out_dir, "fig_selective")


# --------------------------------------------------------------------------- #
# 6. Orientation
# --------------------------------------------------------------------------- #

def fig_orientation(out_dir: Path) -> None:
    runs = _runs()
    sub = runs[(runs["label_mode"] == "soft") & (runs["finetune"] == "full")
               & (runs["loss"] == "bce") & (runs["train_size"] == 0)]
    archs = [a for a in ("resnet50", "convnext_tiny", "vit_small")
             if a in set(sub["arch"])]
    if not archs:
        print("skipping fig_orientation: no runs")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    policies = ["none", "flip", "d4", "d4_photo"]
    x = np.arange(len(policies))
    gap = 0.03
    w = min(BAR_MAX, (0.8 - gap * (len(archs) - 1)) / len(archs))

    for k, arch in enumerate(archs):
        offset = (k - (len(archs) - 1) / 2) * (w + gap)
        vals, errs = [], []
        for policy in policies:
            rows = sub[(sub["arch"] == arch) & (sub["policy"] == policy)
                       & (~sub["orientation_pooled"])]
            vals.append(rows["test_accuracy"].mean() if not rows.empty else np.nan)
            errs.append(rows["test_accuracy"].std(ddof=1) if len(rows) > 1 else 0.0)
        # one colour per architecture, never a shade of one colour: a lightness
        # ramp on nominal categories double-encodes and fails the colour checks
        _bars(axes[0], x + offset, vals, PROBE_COLOUR[arch], w, errs=errs,
              label=PRETTY_ARCH.get(arch, arch))
        if PROBE_COLOUR[arch] == NEEDS_LABEL and np.isfinite(vals[-1]):
            axes[0].annotate(PRETTY_ARCH.get(arch, arch),
                             (x[-1] + offset, vals[-1]), textcoords="offset points",
                             xytext=(0, 4), ha="center", fontsize=6.5,
                             color=INK_SECOND)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([PRETTY_POLICY[p] for p in policies], rotation=15)
    _tidy(axes[0], None, "test accuracy", "(a) augmentation policy")
    finite = sub["test_accuracy"].dropna()
    if len(finite):
        axes[0].set_ylim(max(0.5, finite.min() - 0.03), finite.max() + 0.012)
    axes[0].legend(loc="lower right", fontsize=7)

    # The architecture is carried by colour and by the legend already, so the tick
    # labels only need the condition. Repeating the model name on every tick makes
    # the axis unreadable and says nothing the colour has not said.
    ax = axes[1]
    conditions = (("none", False, "none"), ("d4", False, r"$D_4$"),
                  ("none", True, "built in"))
    labels, values, colours, positions = [], [], [], []
    pos = 0.0
    for arch in archs:
        for policy, pooled, tag in conditions:
            rows = sub[(sub["arch"] == arch) & (sub["policy"] == policy)
                       & (sub["orientation_pooled"] == pooled)]
            if rows.empty:
                continue
            labels.append(tag)
            values.append(rows["d4_invariance_error"].mean())
            colours.append(PROBE_COLOUR[arch])
            positions.append(pos)
            pos += 1.0
        pos += 0.6                       # a gap of surface between architectures
    ax.bar(positions, values, width=0.7, color=colours)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    _tidy(ax, None, r"$\varepsilon_{D_4}$", "(b) how far from invariant")
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

    # the Pareto front: nothing above and to the right of these
    ordered = g.sort_values("throughput", ascending=False)
    best, front_names, xs, ys = -np.inf, [], [], []
    for arch, row in ordered.iterrows():
        if row["accuracy"] > best:
            best = row["accuracy"]
            front_names.append(arch)
            xs.append(row["throughput"]); ys.append(row["accuracy"])

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.step(xs, ys, where="post", lw=0.9, color=INK_MUTED, ls=(0, (4, 3)), zorder=2,
            label="Pareto front")
    for arch, row in g.iterrows():
        ax.scatter(row["throughput"], row["accuracy"],
                   s=24 + 110 * np.sqrt(row["params"] / g["params"].max()),
                   color=FAMILY_COLOUR[family_of(arch)],
                   edgecolor=SURFACE, linewidth=RING_W, zorder=3)
        # Split the labels into two columns -- front models to the right, the rest
        # to the left -- so that clusters of similar models do not overprint each
        # other. Which side a label sits on is itself informative.
        on_front = arch in front_names
        ax.annotate(PRETTY_ARCH.get(arch, arch), (row["throughput"], row["accuracy"]),
                    textcoords="offset points",
                    xytext=(8, 2) if on_front else (-8, 2),
                    ha="left" if on_front else "right",
                    fontsize=6.5, color=INK if on_front else INK_SECOND)

    ax.set_xscale("log")
    ax.margins(x=0.28)
    _tidy(ax, "inference throughput (images/s, one H100)", "test accuracy")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=6, color=c,
                          markeredgecolor=SURFACE, markeredgewidth=RING_W,
                          label=PRETTY_FAMILY[f])
               for f, c in FAMILY_COLOUR.items()]
    handles.append(plt.Line2D([], [], ls=(0, (4, 3)), lw=0.9, color=INK_MUTED,
                              label="Pareto front"))
    ax.legend(handles=handles, loc="lower left", fontsize=7)
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
    xcol = "test_balanced_accuracy_tuned" if "test_balanced_accuracy_tuned" in sub \
        else "test_balanced_accuracy"
    ycol = "decals_balanced_accuracy_tuned" if "decals_balanced_accuracy_tuned" in sub \
        else "decals_balanced_accuracy"

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for mode, marker in (("hard", "s"), ("soft", "o")):
        part = sub[sub["label_mode"] == mode]
        if part.empty:
            continue
        g = part.groupby("arch")[[xcol, ycol]].mean()
        ax.scatter(g[xcol], g[ycol], marker=marker, s=34, color=MODE_COLOUR[mode],
                   edgecolor=SURFACE, linewidth=RING_W, label=f"{mode} labels",
                   zorder=3)
        if mode == "soft":
            for arch, row in g.iterrows():
                ax.annotate(PRETTY_ARCH.get(arch, arch), (row[xcol], row[ycol]),
                            textcoords="offset points", xytext=(6, -2), fontsize=6,
                            color=INK_SECOND)
    lo = float(min(sub[ycol].min(), sub[xcol].min())) - 0.02
    hi = float(max(sub[ycol].max(), sub[xcol].max())) + 0.02
    ax.plot([lo, hi], [lo, hi], lw=0.8, color=INK_MUTED, ls=(0, (4, 3)))
    ax.annotate("no loss in transfer", (hi, hi), textcoords="offset points",
                xytext=(-4, -12), ha="right", fontsize=6.5, color=INK_SECOND)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    _tidy(ax, "balanced accuracy on SDSS (Galaxy Zoo 2)",
          "balanced accuracy on DECaLS (Galaxy10)")
    ax.legend(loc="upper left", fontsize=7)
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
    excess = "background_excess" if "background_excess" in merged else "background_reliance"

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))

    ax = axes[0]
    for family, colour in FAMILY_COLOUR.items():
        part = merged[merged["arch"].map(family_of) == family]
        if part.empty:
            continue
        ax.scatter(part["test_accuracy"], part[excess], s=30, color=colour,
                   edgecolor=SURFACE, linewidth=RING_W, label=PRETTY_FAMILY[family],
                   zorder=3)
    # Label selectively. With one point per explained run the panel would be
    # unreadable if every one were named, so we name the extremes -- which is where
    # the story is -- and let the legend and Table 6 carry the rest.
    ranked = merged.dropna(subset=[excess]).sort_values(excess)
    named = pd.concat([ranked.head(2), ranked.tail(2),
                       ranked[ranked[excess] > 0]]).drop_duplicates("run_id")
    for k, (_, row) in enumerate(named.iterrows()):
        ax.annotate(f"{PRETTY_ARCH.get(row['arch'], row['arch'])} ({row['label_mode']})",
                    (row["test_accuracy"], row[excess]), textcoords="offset points",
                    xytext=(7, 3) if k % 2 == 0 else (-7, -9),
                    ha="left" if k % 2 == 0 else "right",
                    fontsize=6.5, color=INK)
    ax.axhline(0.0, lw=0.8, color=INK_MUTED, ls=(0, (4, 3)))
    ax.annotate("a uniform map scores here", (ax.get_xlim()[0], 0.0),
                textcoords="offset points", xytext=(3, 3), fontsize=6.5,
                color=INK_MUTED)
    ax.margins(x=0.22)
    _tidy(ax, "test accuracy", "attribution outside the galaxy,\nexcess over uniform",
          "(a) accuracy is not groundedness")

    ax = axes[1]
    hi_col = f"{excess}_high_agreement"
    lo_col = f"{excess}_low_agreement"
    if hi_col not in merged or lo_col not in merged:
        ax.axis("off")
    else:
        # One row per architecture rather than per run: with both label modes and
        # the pooled variants there are too many rows to read at this size, and the
        # hard-label values are in Table 6 anyway.
        panel = merged[merged["label_mode"] == "soft"]
        if panel.empty:
            panel = merged
        panel = panel.drop_duplicates("arch").sort_values(excess)
        y = np.arange(len(panel))
        gap, h = 0.08, 0.36
        ax.barh(y - (h + gap) / 2, panel[hi_col], height=h, color=ORDINAL[0],
                label="volunteers agreed")
        ax.barh(y + (h + gap) / 2, panel[lo_col], height=h, color=ORDINAL[1],
                label="volunteers disagreed")
        ax.axvline(0.0, lw=0.8, color=INK_MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels([PRETTY_ARCH.get(a, a) for a in panel["arch"]], fontsize=6.5)
        _tidy(ax, "excess over uniform", None, "(b) soft-label models, by agreement")
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=7)

    # Each panel keeps its own legend -- the two encode different things, and one
    # shared key would have to claim a colour means the same in both -- and both sit
    # in the title row, where no data can collide with them whatever the values.
    axes[0].legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, fontsize=7)
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

    fig, ax = plt.subplots(figsize=(6.6, 2.4 + 0.16 * len(ranks)))
    ax.set_xlim(0.4, len(ranks) + 0.6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.80, 1, len(ranks), color=AXIS, lw=1.0)
    for tick in range(1, len(ranks) + 1):
        ax.vlines(tick, 0.78, 0.82, color=AXIS, lw=1.0)
        ax.text(tick, 0.85, str(tick), ha="center", fontsize=7, color=INK_MUTED)
    ax.text(len(ranks) / 2 + 0.5, 0.93, "mean rank (1 = best)", ha="center",
            fontsize=8, color=INK_SECOND)

    ax.hlines(0.70, 1, 1 + cd, color=INK, lw=1.5)
    ax.vlines([1, 1 + cd], 0.68, 0.72, color=INK, lw=1.5)
    ax.text(1 + cd / 2, 0.62, f"CD = {cd:.2f}", ha="center", fontsize=7,
            color=INK_SECOND)

    step = 0.52 / max(1, len(ranks))
    for i, (arch, rank) in enumerate(ranks.items()):
        y = 0.52 - i * step
        side = -1 if i < len(ranks) / 2 else 1
        anchor = 0.8 if side < 0 else len(ranks) + 0.2
        colour = FAMILY_COLOUR[family_of(arch)]
        ax.plot([rank, rank, anchor], [0.78, y, y], color=INK_MUTED, lw=0.7)
        ax.plot([rank], [0.78], marker="o", ms=4, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=1.0)
        ax.text(anchor + 0.12 * side, y,
                f"{PRETTY_ARCH.get(arch, arch)} ({rank:.2f})", va="center",
                ha="left" if side > 0 else "right", fontsize=7, color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=5, color=c,
                          markeredgecolor=SURFACE, label=PRETTY_FAMILY[f])
               for f, c in FAMILY_COLOUR.items()]
    ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
              bbox_to_anchor=(0.5, -0.04))
    _save(fig, out_dir, "fig_critical_difference")


# --------------------------------------------------------------------------- #
# 11. The protocol, drawn
# --------------------------------------------------------------------------- #

def fig_pipeline(out_dir: Path) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    # tints of the two leading hues plus a neutral, so the schematic reads as one
    # family with the rest of the figures rather than as decoration
    DATA = "#e4eefc"
    MODEL = "#fce8de"
    NEUTRAL = "#f0efec"
    RESULT = "#e0f4ec"

    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.grid(False)

    boxes = [
        (0.1, 1.15, 2.0, "Galaxy Zoo 2\n(SDSS cutouts +\nvolunteer votes)", DATA),
        (2.4, 1.15, 1.9, "task-01 vote\nfraction $p$,\npanel size $N$", DATA),
        (4.6, 2.05, 2.2, "training target:\nhard $1[p>0.5]$\nor soft $p$", MODEL),
        (4.6, 0.25, 2.2, "vote model:\n$\\pi(p,N)$ and the\nachievable ceiling", NEUTRAL),
        (7.1, 1.15, 2.1, "12 backbones\n$D_4$ augmentation\nor built-in\ninvariance", MODEL),
        (9.5, 1.15, 2.3, "agreement-resolved\nerror, calibration,\nselective prediction,\nexplanations", RESULT),
    ]
    for x, y, w, text, colour in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, 1.15, boxstyle="round,pad=0.06",
                                    linewidth=0, facecolor=colour))
        ax.text(x + w / 2, y + 0.575, text, ha="center", va="center", fontsize=6.8,
                color=INK)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.9, color=INK_MUTED))

    arrow(2.1, 1.73, 2.4, 1.73)
    arrow(4.3, 1.73, 4.6, 2.63)
    arrow(4.3, 1.73, 4.6, 0.83)
    arrow(6.8, 2.63, 7.1, 1.9)
    arrow(9.2, 1.73, 9.5, 1.73)
    arrow(6.8, 0.83, 9.5, 1.4)
    ax.text(11.8, 0.15, "held out: Galaxy10 DECaLS", fontsize=6.5, ha="right",
            style="italic", color=INK_SECOND)
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
