"""Paper figures, drawn from the tables in $GZM_WORK/results.

    python -m src.figures --out /path/to/paper5
    python -m src.figures --out /path/to/paper5 --only agreement curve

Vector PDF, one file each, embedded fonts. Missing inputs skip a figure with a note
instead of killing the run, so this is safe to call while the sweep is going.

Colour rules, so they do not drift: three hues, fixed order, never cycled. They mean
architecture family or training target and the legend always says which. Three is the
most that stays distinguishable under colour-blind simulation when any pair can meet,
which happens in the scatter plots. More distinctions go in more panels. The green is
too light to read as text and too low-contrast to stand alone, so anything drawn in it
also gets a direct label. No dual y-axes anywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import config
from src.common import ensure_dir, load_table, read_json, risk_coverage_curve
from src.registry import ARCHITECTURES, PRETTY_ARCH, PRETTY_FAMILY, REGISTRY, family_of
from src.tables import PRETTY_POLICY, reference

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

# checked under deuteranopia and protanopia simulation with every pair together
SLOT = ("#2a78d6", "#eb6834", "#1baf7a")   # blue, orange, green

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECOND = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# an entity keeps its colour in every figure it appears in
FAMILY_COLOUR = {"custom": SLOT[0], "cnn": SLOT[1], "transformer": SLOT[2]}
MODE_COLOUR = {"hard": SLOT[0], "soft": SLOT[1],
               "hard_conf": SLOT[2], "soft_w": SLOT[2]}
PROBE_COLOUR = {"resnet50": SLOT[0], "convnext_tiny": SLOT[1], "vit_small": SLOT[2]}

NEEDS_LABEL = SLOT[2]      # too low-contrast on white to stand without a label

# agreed against contested is an ordered contrast, not a categorical one, so it uses
# two steps of one hue rather than two more categorical colours
ORDINAL = ("#86b6ef", "#1c5cab")

LINE_W = 1.5
MARKER = 5.0
RING_W = 1.2
FILL_ALPHA = 0.10
BAR_MAX = 0.24

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


def _place_labels(fig, ax, items, fontsize=6.5, pad=3.0, marker_radius=6.0):
    """Label points so that labels overlap neither each other nor any marker.

    Fixed offsets, or offsets alternating on a cycle, are enough while the points are
    spread out and fail as soon as several land close together, which is exactly what
    happens when half a dozen architectures perform alike. For each point in turn this
    tries six positions around the marker and takes the first that collides with
    nothing already on the canvas, meaning earlier labels and every marker including
    the ones not yet labelled, falling back to the first position if all six collide.
    The order is the caller's, so the most important labels choose first.

    items: iterable of (x, y, text, colour) in data coordinates.
    """
    fig.canvas.draw()                      # transData is only valid once laid out
    scale = fig.dpi / 72.0                 # offsets are quoted in points
    items = list(items)
    # three rings outward. A label in the first ring sits against its marker and needs
    # nothing else; one pushed to an outer ring is far enough that ownership stops
    # being obvious, so it gets a leader line back to its point.
    ring1 = ((9, -3), (-9, -3), (7, 5), (-7, 5), (7, -12), (-7, -12))
    ring2 = ((17, 9), (-17, 9), (17, -17), (-17, -17), (21, -3), (-21, -3))
    ring3 = ((27, 19), (-27, 19), (27, -27), (-27, -27), (5, 20), (5, -26))
    places = ring1 + ring2 + ring3
    r = marker_radius * scale
    # every marker is an obstacle from the start, so a label never lands on a point
    taken = [(px - r, py - r, px + r, py + r)
             for px, py in (ax.transData.transform((x, y)) for x, y, _, _ in items)]

    def box_at(px, py, dx, dy, w, h):
        left = px + dx * scale if dx > 0 else px + dx * scale - w
        bottom = py + dy * scale
        return (left - pad, bottom - pad, left + w + pad, bottom + h + pad)

    for x, y, text, colour in items:
        px, py = ax.transData.transform((x, y))
        w = 0.58 * fontsize * len(text) * scale     # a serviceable width estimate
        h = 1.25 * fontsize * scale
        dx, dy = next(
            (p for p in places
             if all(box_at(px, py, *p, w, h)[2] < o[0] or box_at(px, py, *p, w, h)[0] > o[2]
                    or box_at(px, py, *p, w, h)[3] < o[1]
                    or box_at(px, py, *p, w, h)[1] > o[3] for o in taken)),
            places[0])
        taken.append(box_at(px, py, dx, dy, w, h))
        leader = dict(arrowprops=dict(arrowstyle="-", lw=0.5, color=INK_MUTED,
                                      shrinkA=0, shrinkB=3)) \
            if (dx, dy) not in ring1 else {}
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha="left" if dx > 0 else "right", fontsize=fontsize, color=colour,
                    **leader)


def _headroom(ax, fraction: float) -> None:
    """Open empty space at the top of the axes so a legend can sit inside it.

    Legends anchored outside the axes collide with the panel title, and loc="best"
    moves with the data, which makes the layout unreproducible. Reserving a band and
    putting the legend in it is stable whatever the values turn out to be.
    """
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * fraction)


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
    # Prefer the working arrays when they are there, since they are the source of
    # truth, and fall back to the published sample when they are not. The sample is a
    # few megabytes in results/ and it exists so that this figure survives the cluster
    # storage being reclaimed: without it the only figure in the paper that cannot be
    # redrawn from the repository is this one.
    sample_path = config.RESULTS / "dataset_sample.npz"
    if (config.ARRAYS / "gz2_table.csv").exists():
        table = load_table("gz2")
        images = np.load(config.ARRAYS / "gz2_images.npy", mmap_mode="r")
        picked = None
    elif sample_path.exists():
        s = np.load(sample_path, allow_pickle=False)
        table = pd.DataFrame({"p_featured": s["p_featured"],
                              "agreement": s["agreement"],
                              "votes": s["votes"]})
        images = s["sample_images"]
        picked = pd.DataFrame({"bin": s["sample_bin"], "p_featured": s["sample_p"],
                               "agreement": s["sample_agreement"],
                               "votes": s["sample_votes"],
                               "row": np.arange(len(s["sample_bin"]))})
    else:
        print("skipping fig_dataset: neither the arrays nor dataset_sample.npz")
        return

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.0], hspace=0.34, wspace=0.32)

    # one series per panel, so no legend: the axis label says what is plotted
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(table["p_featured"], bins=60, color=SLOT[0])
    _tidy(ax, r"vote fraction $p$ (featured)", "galaxies", "(a) volunteer votes")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(table["agreement"], bins=50, color=SLOT[0])
    ax.axvline(0.6, color=INK_MUTED, lw=0.8)
    # the histogram rises steeply to the right of the threshold, so the label goes on
    # the left of the line where the bars are low
    ax.annotate("confident\nthreshold", xy=(0.6, ax.get_ylim()[1]),
                xytext=(-4, -1), textcoords="offset points", fontsize=6.5,
                color=INK_SECOND, va="top", ha="right")
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
    ax.spines["left"].set_visible(False)   # no y scale, so no y spine

    # A single example per bin invited the reading that the contested galaxies are
    # simply odd objects. Three of them, drawn independently, show that the whole bin
    # looks like that, which is the claim the rest of the paper rests on. Columns are
    # agreement bins and rows are independent draws, so reading down a column shows
    # the variety inside one bin and reading across a row shows the progression.
    rng = np.random.default_rng(config.SEED)
    n_rows = 3
    columns = []
    for b in range(len(edges) - 1):
        if picked is not None:
            chosen = picked[picked["bin"] == b]
            if len(chosen):
                columns.append((b, chosen))
            continue
        sub = table[(table["agreement"] >= edges[b])
                    & (table["agreement"] < edges[b + 1] + 1e-9)]
        if len(sub):
            take = rng.choice(len(sub), size=min(n_rows, len(sub)), replace=False)
            columns.append((b, sub.iloc[take]))
    inner = gs[1, :].subgridspec(n_rows, len(columns), wspace=0.06, hspace=0.30)
    for col, (b, chosen) in enumerate(columns):
        for r in range(n_rows):
            ax = fig.add_subplot(inner[r, col])
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r >= len(chosen):
                ax.set_visible(False)
                continue
            row = chosen.iloc[r]
            ax.imshow(np.asarray(images[int(row["row"])]))
            ax.set_title(f"$p$={row['p_featured']:.2f}", fontsize=6,
                         color=INK_MUTED, pad=2)
            if r == 0:      # the bin heading sits above the first row of its column
                ax.annotate(f"$a \\in [{edges[b]:.1f}, {edges[b + 1]:.1f})$",
                            (0.5, 1.0), xycoords="axes fraction",
                            textcoords="offset points", xytext=(0, 16),
                            ha="center", fontsize=6.5, color=INK_SECOND)
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

    sub = reference(agr)
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
                            textcoords="offset points", xytext=(14, 5),
                            ha="left", fontsize=6.5, color=INK_SECOND)

        if "majority_baseline" in part:
            base = part.groupby("bin")["majority_baseline"].mean()
            ax.plot(_bin_centres(base.index), base.to_numpy(), lw=1.0,
                    color=INK_MUTED, label="majority baseline in bin")

        # The ceiling is a property of the vote distribution, and the bins are cut on
        # the vote distribution, so a single global value drawn straight across is the
        # wrong comparison: the near-unanimous bin rises above it and reads as a
        # violation, the contested bin sits far below it and reads as headroom. Draw
        # it per bin where that is available and keep the global value only as a
        # fallback for older result files.
        per_bin = (ceiling or {}).get("by_agreement_bin") or {}
        if per_bin:
            bins = sorted(int(b) for b in per_bin)
            xs = _bin_centres(pd.Index(bins))
            ys = [per_bin[str(b)]["bayes_accuracy"] for b in bins]
            ax.plot(xs, ys, lw=1.0, color=INK_MUTED, ls=(0, (4, 3)),
                    label=r"$A^\star$ in bin")
        elif ceiling:
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
    # train_size is the free axis, so everything else is pinned by hand, including
    # the input size: without it the full-split end of each curve averages in the
    # resolution-sweep runs and the curve is read against the ceiling from a point
    # that no single configuration produced
    native = runs["arch"].map(lambda a: REGISTRY[a].input_size if a in REGISTRY else None)
    sub = runs[(runs["policy"] == "d4") & (runs["finetune"] == "full")
               & (runs["loss"] == "bce") & (~runs["orientation_pooled"])
               & (runs["pretrained"]) & (runs["size"] == native)]
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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), layout="constrained")

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
    tallest = 0.0
    for k, (mode, column, label) in enumerate([
            ("hard", "test_ece", "hard, as measured"),
            ("soft", "test_ece", "soft, as measured")]):
        vals = [ref[(ref["arch"] == a) & (ref["label_mode"] == mode)][column].mean()
                for a in archs]
        errs = [ref[(ref["arch"] == a) & (ref["label_mode"] == mode)][column].std(ddof=1)
                for a in archs]
        _bars(ax, x + (k - 0.5) * (w + gap), vals, MODE_COLOUR[mode], w,
              errs=errs, label=label)
        tallest = max(tallest, float(np.nanmax(np.add(vals, np.nan_to_num(errs)))))
    if "test_ece_calibrated" in ref:
        vals = [ref[ref["arch"] == a]["test_ece_calibrated"].mean() for a in archs]
        ax.plot(x, vals, ls="none", marker="_", ms=12, color=INK,
                markeredgewidth=1.4, label="either, after temperature scaling")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_ARCH.get(a, a) for a in archs], rotation=60,
                       ha="right", fontsize=6.5)
    _tidy(ax, None, "expected calibration error", "(b) by architecture")
    # The soft bars all sit near 0.14 and the three-entry legend needs a clear band
    # above them. Rounding the limit up to the next fortieth leaves that band without
    # cropping anything, whatever the values turn out to be.
    ax.set_ylim(0, np.ceil(max(tallest, 0.14) / 0.025) * 0.025 + 0.05)
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

    curves = _optional("risk_coverage.csv")
    if curves is None or curves.empty:
        print("skipping fig_selective: risk_coverage.csv missing; rerun src.analysis "
              "where the per-run predictions live")
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
    lo = 1.0
    for arch, mode, run_id in picks:
        part = curves[curves["run_id"] == run_id].sort_values("coverage")
        if part.empty:
            continue
        ax.plot(part["coverage"], part["accuracy"], lw=LINE_W,
                ls="-" if mode == "soft" else (0, (4, 2)),
                color=FAMILY_COLOUR[family_of(arch)],
                label=f"{PRETTY_ARCH.get(arch, arch)}, {mode}",
                solid_capstyle="round")
        lo = min(lo, float(part["accuracy"].min()))
    for target in (0.99, 0.95):
        ax.axhline(target, lw=0.7, color=INK_MUTED, ls=(0, (2, 3)))
        # left end, where the curves are still flat against the top of the frame and
        # nothing is in the way; at the right end they are falling through the line
        ax.annotate(f"{target:.2f}", (0.06, target), textcoords="offset points",
                    xytext=(0, 3), ha="left", fontsize=6.5, color=INK_MUTED)
    _tidy(ax, "coverage (fraction classified automatically)",
          "accuracy on the covered part")
    # The whole trade-off happens in the top few points of the scale. Drawn from zero
    # the curves are two flat lines against the top of the frame and the figure says
    # nothing; the range has to be the range the data occupies.
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(min(lo, 0.94) - 0.004, 1.002)
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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
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
        # one colour per architecture, not shades of one: these are nominal
        _bars(axes[0], x + offset, vals, PROBE_COLOUR[arch], w, errs=errs,
              label=PRETTY_ARCH.get(arch, arch))
        if PROBE_COLOUR[arch] == NEEDS_LABEL and np.isfinite(vals[-1]):
            axes[0].annotate(PRETTY_ARCH.get(arch, arch),
                             (x[-1] + offset, vals[-1]), textcoords="offset points",
                             xytext=(0, 4), ha="center", fontsize=6.5,
                             color=INK_SECOND)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([PRETTY_POLICY[p] for p in policies], rotation=20,
                            ha="right")
    _tidy(axes[0], None, "test accuracy", "(a) augmentation policy")
    finite = sub["test_accuracy"].dropna()
    if len(finite):
        axes[0].set_ylim(max(0.5, finite.min() - 0.03), finite.max() + 0.012)
    _headroom(axes[0], 0.34)
    axes[0].legend(loc="upper left", fontsize=7)

    # The architecture is carried by colour and by the legend already, so the tick
    # labels only need the condition. Repeating the model name on every tick makes
    # the axis unreadable and says nothing the colour has not said.
    ax = axes[1]
    conditions = (("none", False, "none"), ("d4", False, r"$D_4$"),
                  ("none", True, "pooled"))
    labels, values, colours, positions = [], [], [], []
    groups = []                          # (centre, architecture) for the group labels
    pos = 0.0
    for arch in archs:
        start = pos
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
        if pos > start:
            groups.append(((start + pos - 1.0) / 2, arch))
        pos += 0.8                       # a gap of surface between architectures
    ax.bar(positions, values, width=0.7, color=colours)
    # The pooled bars are zero up to float round-off, so they draw nothing and the
    # reader is left to infer an absence. Mark them, since they are the panel's whole
    # point, and give the order of magnitude rather than writing a bare 0 that the
    # measurement does not support.
    tallest = max((v for v in values if v is not None and np.isfinite(v)), default=1.0)
    for p, v in zip(positions, values):
        if v is not None and np.isfinite(v) and v < 0.01 * tallest:
            ax.annotate(f"$10^{{{int(np.floor(np.log10(max(v, 1e-30))))}}}$", (p, 0.0),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=6, color=INK_SECOND)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    _tidy(ax, None, r"$\varepsilon_{D_4}$", "(b) how far from invariant")
    for centre, arch in groups:          # architecture under its own group of bars
        ax.annotate(PRETTY_ARCH.get(arch, arch), (centre, 0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, -30), ha="center",
                    fontsize=6.5, color=INK_SECOND, annotation_clip=False)
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

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.step(xs, ys, where="post", lw=0.9, color=INK_MUTED, ls=(0, (4, 3)), zorder=2,
            label="Pareto front")
    for arch, row in g.iterrows():
        ax.scatter(row["throughput"], row["accuracy"],
                   s=24 + 110 * np.sqrt(row["params"] / g["params"].max()),
                   color=FAMILY_COLOUR[family_of(arch)],
                   edgecolor=SURFACE, linewidth=RING_W, zorder=3)

    ax.set_xscale("log")
    ax.margins(x=0.28)
    _tidy(ax, "inference throughput (images/s, one H100)", "test accuracy")
    # Front models are named in full ink and get first choice of label position, since
    # they are the ones a reader looks for; the rest are recessive. Half the field sits
    # within a factor of two in throughput and a fifth of a point in accuracy, so the
    # positions have to be decluttered rather than assigned from a fixed cycle.
    ordered_labels = ([a for a in front_names]
                      + [a for a in g.sort_values("accuracy", ascending=False).index
                         if a not in front_names])
    _place_labels(fig, ax, [(g.loc[a, "throughput"], g.loc[a, "accuracy"],
                             PRETTY_ARCH.get(a, a),
                             INK if a in front_names else INK_SECOND)
                            for a in ordered_labels])
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

    means = {mode: sub[sub["label_mode"] == mode].groupby("arch")[[xcol, ycol]].mean()
             for mode in ("hard", "soft") if (sub["label_mode"] == mode).any()}
    if not means:
        print("skipping fig_cross_survey: no runs")
        return

    # The quantity of interest is the paired difference, so plot it directly. A
    # source-versus-target scatter against the diagonal is the conventional form, but
    # it does not survive this distribution: ten of the twelve architectures sit
    # within two points of no change, six of them on the gaining side, while two
    # lose between three and thirty-five, so any shared square range collapses the
    # ten into an illegible corner. Penalty per architecture puts each on its own
    # labelled row, and because the values leave a genuine empty gap of fifteen
    # points the x axis is broken across that gap, drawn as such, so the near-zero
    # region and the collapse are both readable at their own scale. Absolute levels
    # are in the accompanying table.
    penalty = {mode: ((g[xcol] - g[ycol]) * 100).rename("penalty")
               for mode, g in means.items()}
    order = (penalty.get("soft", next(iter(penalty.values())))).sort_values()
    y = {arch: k for k, arch in enumerate(order.index)}
    values = np.concatenate([s.to_numpy() for s in penalty.values()])

    # Where to break: the upper Tukey fence, so the left panel holds the bulk at its
    # own scale and only genuine outliers go to the right panel. A fixed rule rather
    # than the widest observed gap, because the widest gap moves with one architecture
    # and would silently change the figure's meaning between runs.
    ordered_vals = np.sort(values)
    q1, q3 = np.percentile(values, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    outliers = ordered_vals[ordered_vals > fence]
    split = len(outliers) > 0 and len(outliers) < len(values) / 3
    if split:
        bulk_max = float(ordered_vals[ordered_vals <= fence].max())
        break_lo, break_hi = bulk_max, float(outliers.min())
    ratios = [3, 1] if split else [1]
    fig, axs = plt.subplots(1, len(ratios), figsize=(5.8, 3.9), sharey=True,
                            gridspec_kw={"width_ratios": ratios, "wspace": 0.06},
                            layout="constrained")
    axs = np.atleast_1d(axs)

    for ax in axs:
        if len(penalty) == 2:      # hairline pairing the two modes of one architecture
            a, b = penalty.values()
            for arch, row in y.items():
                if arch in a.index and arch in b.index:
                    ax.plot([float(a[arch]), float(b[arch])], [row, row], lw=0.7,
                            color=GRID, zorder=1)
        for mode, marker in (("hard", "s"), ("soft", "o")):
            series = penalty.get(mode)
            if series is None:
                continue
            keep = [a for a in series.index if a in y]
            ax.scatter([float(series[a]) for a in keep], [y[a] for a in keep],
                       marker=marker, s=34, color=MODE_COLOUR[mode],
                       edgecolor=SURFACE, linewidth=RING_W, label=f"{mode} labels",
                       zorder=3)
        ax.axvline(0.0, lw=0.8, color=INK_MUTED, zorder=2)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.7, len(y) - 0.3)

    if split:
        left_pad = 0.10 * (break_lo - ordered_vals[0])
        right_pad = 0.12 * max(1.0, ordered_vals[-1] - break_hi)
        axs[0].set_xlim(ordered_vals[0] - left_pad, break_lo + left_pad)
        axs[1].set_xlim(break_hi - right_pad, ordered_vals[-1] + right_pad)
        axs[0].spines["right"].set_visible(False)
        axs[1].spines["left"].set_visible(False)
        axs[1].tick_params(axis="y", length=0)
        # the smallest outlier sits at the right panel's left edge, where the automatic
        # locator puts no tick, so it gets one: otherwise that marker has no scale
        auto = [t for t in axs[1].get_xticks() if break_hi + right_pad < t
                <= ordered_vals[-1] + right_pad]
        axs[1].set_xticks(sorted({round(break_hi)} | set(auto)))
        for ax, xpos in ((axs[0], 1.0), (axs[1], 0.0)):   # the break marks
            ax.plot([xpos, xpos], [0, 1], transform=ax.transAxes, clip_on=False,
                    marker=[(-1, -2.2), (1, 2.2)], ms=5, mew=0.9,
                    color=AXIS, ls="none", zorder=5)
        axs[0].set_title(f"{len(values) - len(outliers)} of {len(values)} runs",
                         fontsize=6.5, color=INK_SECOND, loc="left")
        axs[1].set_title("outliers, own scale", fontsize=6.5, color=INK_SECOND,
                         loc="left")

    axs[0].set_yticks(range(len(y)))
    axs[0].set_yticklabels([PRETTY_ARCH.get(a, a) for a in order.index], fontsize=7)
    axs[0].annotate("gains", (0.0, -0.55), textcoords="offset points",
                    xytext=(-4, 0), ha="right", va="center", fontsize=6.5,
                    color=INK_MUTED)
    axs[0].annotate("loses", (0.0, -0.55), textcoords="offset points",
                    xytext=(4, 0), ha="left", va="center", fontsize=6.5,
                    color=INK_MUTED)
    axs[0].legend(loc="upper left", fontsize=7)
    for ax in axs:
        _tidy(ax)
    fig.supxlabel("balanced accuracy lost in transfer (percentage points)",
                  fontsize=9, color=INK)
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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), layout="constrained")

    ax = axes[0]
    for family, colour in FAMILY_COLOUR.items():
        part = merged[merged["arch"].map(family_of) == family]
        if part.empty:
            continue
        ax.scatter(part["test_accuracy"], part[excess], s=30, color=colour,
                   edgecolor=SURFACE, linewidth=RING_W, label=PRETTY_FAMILY[family],
                   zorder=3)
    # Label selectively. With one point per explained run the panel would be
    # unreadable if every one were named, so we name the extremes, which is where
    # the story is, and let the legend and Table 6 carry the rest.
    ranked = merged.dropna(subset=[excess]).sort_values(excess)
    named = pd.concat([ranked.head(2), ranked.tail(2),
                       ranked[ranked[excess] > 0]]).drop_duplicates("run_id")
    # four offsets rather than two: the named runs include near-duplicate pairs, the
    # same architecture under both label modes, which two alternating offsets still
    # print on top of each other
    corners = ((7, 4, "left"), (-7, -10, "right"), (7, -10, "left"), (-7, 4, "right"))
    for k, (_, row) in enumerate(named.iterrows()):
        dx, dy, ha = corners[k % len(corners)]
        ax.annotate(f"{PRETTY_ARCH.get(row['arch'], row['arch'])} ({row['label_mode']})",
                    (row["test_accuracy"], row[excess]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=6.5, color=INK)
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
        # Nearly every value is negative, so the bars run left from zero and the
        # axis-edge tick labels end up underneath the bar ends. The names therefore go
        # in the free space to the right of the longest positive bar, which is a column
        # no bar can reach whatever the values turn out to be.
        ax.set_yticks(y)
        ax.tick_params(axis="y", labelleft=False, length=0)
        low = float(min(panel[hi_col].min(), panel[lo_col].min(), 0.0))
        high = float(max(panel[hi_col].max(), panel[lo_col].max(), 0.0))
        span = high - low
        label_x = high + 0.03 * span
        for row_y, arch in zip(y, panel["arch"]):
            ax.annotate(PRETTY_ARCH.get(arch, arch), (label_x, row_y),
                        va="center", ha="left", fontsize=6.5, color=INK_SECOND)
        ax.set_xlim(low - 0.04 * span, label_x + 0.40 * span)
        _tidy(ax, "excess over uniform", None, "(b) soft-label models, by agreement")
        ax.spines["left"].set_visible(False)
        _headroom(ax, 0.30)
        ax.legend(loc="upper right", ncol=1, fontsize=7)

    # Each panel keeps its own legend, because the two encode different things and one
    # shared key would have to claim a colour means the same in both. Both sit inside
    # their axes, in headroom opened for them, so neither can reach the panel titles.
    _headroom(axes[0], 0.26)
    axes[0].legend(loc="upper center", ncol=3, fontsize=7)
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

def fig_gradcam(out_dir: Path) -> None:
    path = config.RESULTS / "xai_gallery.npz"
    if not path.exists():
        # a marked placeholder rather than a skip, so the manuscript still builds and
        # the gap is impossible to miss in the PDF
        print("fig_gradcam: xai_gallery.npz missing; writing a placeholder. Rerun "
              "src.xai with --gallery-run <run_id> to fill it in.")
        fig, ax = plt.subplots(figsize=(7.2, 2.0))
        ax.axis("off")
        ax.text(0.5, 0.5, "attribution gallery pending\n(run src.xai --gallery-run)",
                ha="center", va="center", fontsize=11, color="#c0392b")
        _save(fig, out_dir, "fig_gradcam")
        return
    data = np.load(path, allow_pickle=False)
    images, maps, bins = data["images"], data["maps"], data["bin"]
    agreement = data["agreement"]

    edges = np.asarray(config.AGREEMENT_BINS, dtype=float)
    present = sorted(set(int(b) for b in bins))
    per_bin = max(1, int(np.bincount(bins).max()))

    fig, axes = plt.subplots(per_bin, len(present),
                             figsize=(1.42 * len(present), 1.42 * per_bin + 0.5),
                             layout="constrained")
    axes = np.atleast_2d(axes)
    if axes.shape[0] != per_bin:
        axes = axes.T
    for col, b in enumerate(present):
        members = np.flatnonzero(bins == b)
        for r in range(per_bin):
            ax = axes[r, col]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r >= len(members):
                ax.set_visible(False)
                continue
            i = members[r]
            ax.imshow(images[i])
            # Constant alpha does not work on this imagery. The sky is nearly black and
            # so is the low end of any perceptual ramp, so a flat wash is invisible
            # exactly where the attribution is low and muddies the galaxy where it is
            # high. Ramping alpha with the value instead leaves the cutout untouched
            # where the map says nothing and lets the attended region glow, which is
            # what the reader needs to see.
            # The maps are peaked: a few per cent of the pixels carry the top half of
            # the range. A linear alpha therefore shows only the summit and reads as a
            # hard-edged blob, so the ramp is square-rooted to bring the shoulders back.
            m = np.clip(maps[i], 0.0, 1.0)
            im = ax.imshow(m, cmap="inferno", vmin=0.0, vmax=1.0,
                           alpha=np.sqrt(m) * 0.75)
            ax.set_title(f"$a$={agreement[i]:.2f}", fontsize=6, color=INK_MUTED,
                         pad=2)
            if r == 0:
                ax.annotate(f"$a \\in [{edges[b]:.1f}, {edges[b + 1]:.1f})$",
                            (0.5, 1.0), xycoords="axes fraction",
                            textcoords="offset points", xytext=(0, 15),
                            ha="center", fontsize=6.5, color=INK_SECOND)
    bar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.035,
                       pad=0.02, aspect=50)
    bar.set_label("attribution, normalised within each image", fontsize=7,
                  color=INK_SECOND)
    bar.ax.tick_params(labelsize=6, length=0, colors=INK_MUTED)
    bar.outline.set_visible(False)
    _save(fig, out_dir, "fig_gradcam")


FIGURES = {
    "dataset": fig_dataset,
    "agreement": fig_agreement,
    "gradcam": fig_gradcam,
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
