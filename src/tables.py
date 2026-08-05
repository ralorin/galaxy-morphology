"""Write the LaTeX tables and the numeric macros the manuscript uses.

    python -m src.tables --out /path/to/paper5

Produces, in <out>/tables/:

    main.tex           every architecture, hard and soft labels, five seeds
    labels.tex         the four label modes
    orientation.tex    augmentation policies and built-in D4 invariance
    protocol.tex       resolution, fine-tuning depth and loss ablations
    cross_survey.tex   SDSS -> DECaLS transfer
    xai.tex            faithfulness of the explanations
    numbers.tex        \newcommand macros for every figure quoted in the prose

The last file is the important one. Nothing numeric is typed into the manuscript by
hand: the text says \bestSoftAccuracy and this script decides what that is from
results/runs.csv. Rerun the pipeline, rerun this, and the prose cannot drift away
from the experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.common import ensure_dir, read_json
from src.registry import (ARCHITECTURES, PRETTY_ARCH, PRETTY_FAMILY, REGISTRY,
                          family_of)

PRETTY_POLICY = {"none": "none", "flip": "flips", "d4": r"$D_4$", "d4_photo": r"$D_4$ + photometric"}
PRETTY_LABEL = {"hard": "hard", "hard_conf": "hard, confident only",
                "soft": "soft (vote fraction)", "soft_w": "soft, vote-weighted"}


def _load():
    runs = pd.read_csv(config.RESULTS / "runs.csv")
    runs["orientation_pooled"] = runs["orientation_pooled"].fillna(False).astype(bool)
    runs["train_size"] = runs["train_size"].fillna(0).astype(int)
    if "pretrained" not in runs:
        runs["pretrained"] = True
    runs["pretrained"] = runs["pretrained"].fillna(True).astype(bool)
    return runs


def reference(runs: pd.DataFrame) -> pd.DataFrame:
    """The one protocol every headline number comes from: $D_4$ augmentation, full
    fine-tuning from ImageNet, cross-entropy, the whole training split."""
    ref = runs[(runs["policy"] == "d4") & (runs["finetune"] == "full")
               & (runs["loss"] == "bce") & (runs["train_size"] == 0)
               & (~runs["orientation_pooled"])]
    return ref[ref["pretrained"]] if "pretrained" in ref else ref


def _mean_std(frame: pd.DataFrame, column: str, digits: int = 3) -> str:
    if frame.empty or frame[column].isna().all():
        return "--"
    m = frame[column].mean()
    if len(frame) < 2:
        return f"{m:.{digits}f}"
    s = frame[column].std(ddof=1)
    return f"{m:.{digits}f}\\,$\\pm$\\,{s:.{digits}f}"


def _wrap(body: str, caption: str, label: str, colspec: str,
          header: str, small: str = r"\footnotesize") -> str:
    return "\n".join([
        r"\begin{table}[!htb]", r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        small,
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule", header, r"\midrule", body, r"\bottomrule",
        r"\end{tabular}", r"\end{table}", "",
    ])


# --------------------------------------------------------------------------- #
# Main table
# --------------------------------------------------------------------------- #

def table_main(runs: pd.DataFrame, selective: pd.DataFrame | None) -> str:
    ref = reference(runs)
    lines = []
    for family in ("custom", "cnn", "transformer"):
        archs = [a for a in ARCHITECTURES
                 if family_of(a) == family and a in set(ref["arch"])]
        if not archs:
            continue
        lines.append(r"\multicolumn{8}{l}{\textit{" + PRETTY_FAMILY[family] + r"}} \\")
        for arch in archs:
            for mode in ("hard", "soft"):
                rows = ref[(ref["arch"] == arch) & (ref["label_mode"] == mode)]
                if rows.empty:
                    continue
                cov = "--"
                if selective is not None and not selective.empty:
                    sel = selective[selective["run_id"].isin(rows["run_id"])]
                    if not sel.empty:
                        cov = f"{sel['coverage_at_99'].mean():.2f}"
                params = rows["params"].dropna()
                name = PRETTY_ARCH.get(arch, arch)
                lines.append(" & ".join([
                    name if mode == "hard" else "",
                    PRETTY_LABEL[mode].split(" ")[0],
                    _mean_std(rows, "test_accuracy"),
                    _mean_std(rows, "test_balanced_accuracy"),
                    _mean_std(rows, "test_auc_roc"),
                    _mean_std(rows, "test_ece"),
                    cov,
                    f"{params.iloc[0] / 1e6:.1f}" if len(params) else "--",
                ]) + r" \\")
        lines.append(r"\addlinespace")

    header = (r"\textbf{Model} & \textbf{Labels} & \textbf{Accuracy} & "
              r"\textbf{Bal.\ acc.} & \textbf{AUC-ROC} & \textbf{ECE} & "
              r"\textbf{Cov.@99} & \textbf{M par.} \\")
    caption = ("Held-out performance on the Galaxy Zoo~2 test split, mean\\,$\\pm$\\,"
               "standard deviation over five seeds. \\emph{Labels} is the training "
               "target: \\emph{hard} thresholds the volunteer vote fraction, "
               "\\emph{soft} regresses it. \\emph{Cov.@99} is the fraction of the "
               "test set that can be classified automatically while holding 99\\% "
               "accuracy, the rest being deferred. All runs use $D_4$ augmentation, "
               "full fine-tuning and the complete training split.")
    return _wrap("\n".join(lines), caption, "tab:main",
                 "llcccccc", header, r"\scriptsize")


# --------------------------------------------------------------------------- #
# Ablations
# --------------------------------------------------------------------------- #

def table_labels(runs: pd.DataFrame, tracking: pd.DataFrame | None) -> str:
    sub = runs[(runs["policy"] == "d4") & (runs["finetune"] == "full")
               & (runs["train_size"] == 0) & (~runs["orientation_pooled"])
               & (runs["loss"] == "bce")]
    lines = []
    for arch in ("resnet50", "convnext_tiny", "vit_small"):
        rows_arch = sub[sub["arch"] == arch]
        if rows_arch.empty:
            continue
        for i, mode in enumerate(("hard", "hard_conf", "soft", "soft_w")):
            rows = rows_arch[rows_arch["label_mode"] == mode]
            if rows.empty:
                continue
            rho = "--"
            if tracking is not None and not tracking.empty:
                t = tracking[tracking["run_id"].isin(rows["run_id"])]
                if not t.empty:
                    rho = f"{t['spearman_vs_votefraction'].mean():.3f}"
            lines.append(" & ".join([
                PRETTY_ARCH.get(arch, arch) if i == 0 else "",
                PRETTY_LABEL[mode],
                _mean_std(rows, "test_accuracy"),
                _mean_std(rows, "test_auc_roc"),
                _mean_std(rows, "test_ece"),
                _mean_std(rows, "test_brier"),
                rho,
            ]) + r" \\")
        lines.append(r"\addlinespace")

    header = (r"\textbf{Model} & \textbf{Training target} & \textbf{Accuracy} & "
              r"\textbf{AUC-ROC} & \textbf{ECE} & \textbf{Brier} & "
              r"$\boldsymbol{\rho}$ \\")
    caption = ("Effect of the training target. All four modes are scored against the "
               "same thresholded test labels, so the columns are directly comparable. "
               "$\\rho$ is the Spearman correlation between the predicted probability "
               "and the volunteer vote fraction on the test split: it measures whether "
               "the network's confidence recovers human disagreement rather than merely "
               "being well calibrated. \\emph{hard, confident only} trains on the "
               "galaxies with agreement $\\geq 0.6$ and is the control for the "
               "label-noise account.")
    return _wrap("\n".join(lines), caption, "tab:labels", "llccccc", header)


def table_orientation(runs: pd.DataFrame) -> str:
    sub = runs[(runs["label_mode"] == "soft") & (runs["finetune"] == "full")
               & (runs["train_size"] == 0) & (runs["loss"] == "bce")]
    lines = []
    for arch in ("resnet50", "convnext_tiny", "vit_small"):
        first = True
        for policy in ("none", "flip", "d4", "d4_photo"):
            rows = sub[(sub["arch"] == arch) & (sub["policy"] == policy)
                       & (~sub["orientation_pooled"])]
            if rows.empty:
                continue
            lines.append(" & ".join([
                PRETTY_ARCH.get(arch, arch) if first else "",
                PRETTY_POLICY[policy], "learnt",
                _mean_std(rows, "test_accuracy"),
                _mean_std(rows, "test_auc_roc"),
                _mean_std(rows, "d4_invariance_error"),
                f"{rows['train_seconds'].mean() / 60:.0f}",
            ]) + r" \\")
            first = False
        for policy in ("none", "d4"):
            rows = sub[(sub["arch"] == arch) & (sub["policy"] == policy)
                       & (sub["orientation_pooled"])]
            if rows.empty:
                continue
            lines.append(" & ".join([
                "", PRETTY_POLICY[policy], "built in",
                _mean_std(rows, "test_accuracy"),
                _mean_std(rows, "test_auc_roc"),
                _mean_std(rows, "d4_invariance_error"),
                f"{rows['train_seconds'].mean() / 60:.0f}",
            ]) + r" \\")
        lines.append(r"\addlinespace")

    header = (r"\textbf{Model} & \textbf{Augmentation} & \textbf{Invariance} & "
              r"\textbf{Accuracy} & \textbf{AUC-ROC} & "
              r"$\boldsymbol{\varepsilon_{D_4}}$ & \textbf{min} \\")
    caption = ("Orientation. \\emph{learnt} means the network sees rotated and "
               "mirrored copies during training; \\emph{built in} means its logit is "
               "averaged over the eight elements of $D_4$, which makes it exactly "
               "invariant at eight times the inference cost. "
               "$\\varepsilon_{D_4}$ is the mean spread of the predicted probability "
               "over the eight views of a test image, so it is zero for the built-in "
               "models by construction and measures how much of the symmetry the "
               "others failed to absorb. The last column is training wall-clock time.")
    return _wrap("\n".join(lines), caption, "tab:orientation", "lllcccc", header)


def table_protocol(runs: pd.DataFrame) -> str:
    sub = runs[(runs["label_mode"] == "soft") & (runs["train_size"] == 0)
               & (~runs["orientation_pooled"])]
    if "pretrained" not in sub:
        sub = sub.assign(pretrained=True)
    lines = []

    def block(title, frame, knob, order):
        lines.append(r"\multicolumn{4}{l}{\textit{" + title + r"}} \\")
        for arch in ("resnet50", "convnext_tiny", "vit_small"):
            cells = []
            for level in order:
                rows = frame[(frame["arch"] == arch) & (frame[knob] == level)]
                cells.append(_mean_std(rows, "test_accuracy") if not rows.empty else "--")
            while len(cells) < 3:
                cells.append("")
            if all(c in ("--", "") for c in cells):
                continue
            lines.append(" & ".join([PRETTY_ARCH.get(arch, arch), *cells[:3]]) + r" \\")
        lines.append(r"\addlinespace")

    base = sub[(sub["policy"] == "d4") & (sub["loss"] == "bce") & (sub["pretrained"])]
    block("Input resolution: 112 / 160 / 224 px",
          base[base["finetune"] == "full"], "size", [112, 160, 224])
    block("Fine-tuning depth: frozen / last stage / full", base, "finetune",
          ["frozen", "partial", "full"])
    block("Loss: cross-entropy / focal / label smoothing",
          sub[(sub["policy"] == "d4") & (sub["finetune"] == "full") & (sub["pretrained"])],
          "loss", ["bce", "focal", "smooth"])
    block("Initialisation: ImageNet / random",
          sub[(sub["policy"] == "d4") & (sub["finetune"] == "full")
              & (sub["loss"] == "bce")], "pretrained", [True, False])

    header = r"\textbf{Model} & \textbf{level 1} & \textbf{level 2} & \textbf{level 3} \\"
    caption = ("Remaining design choices, test accuracy over two or three seeds. Each "
               "block gives its levels in the order named in the heading. Swin-T is "
               "absent from the resolution block because its window size fixes the "
               "input to 224\\,px. The initialisation block compares ImageNet weights "
               "with random initialisation, in both cases fine-tuning the whole "
               "network; the frozen row of the second block is the linear-probe "
               "setting and should not be confused with it.")
    return _wrap("\n".join(lines), caption, "tab:protocol", "lccc", header)


def table_cross_survey(runs: pd.DataFrame) -> str:
    path = config.RESULTS / "cross_survey.csv"
    if not path.exists():
        return "% cross-survey results not available yet\n"
    cross = pd.read_csv(path)
    cross["orientation_pooled"] = cross["orientation_pooled"].fillna(False).astype(bool)
    lines = []
    for arch in ARCHITECTURES:
        for mode in ("hard", "soft"):
            rows = cross[(cross["arch"] == arch) & (cross["label_mode"] == mode)
                         & (~cross["orientation_pooled"])]
            if rows.empty:
                continue
            lines.append(" & ".join([
                PRETTY_ARCH.get(arch, arch) if mode == "hard" else "",
                mode,
                f"{rows['test_accuracy'].mean():.3f}",
                f"{rows['decals_accuracy'].mean():.3f}",
                f"{rows['decals_balanced_accuracy'].mean():.3f}"
                if "decals_balanced_accuracy" in rows else "--",
                f"{rows['accuracy_drop'].mean():+.3f}",
                f"{rows['decals_auc_roc'].mean():.3f}",
            ]) + r" \\")

    header = (r"\textbf{Model} & \textbf{Labels} & \textbf{GZ2 acc.} & "
              r"\textbf{DECaLS acc.} & \textbf{DECaLS bal.\ acc.} & "
              r"$\boldsymbol{\Delta}$ & \textbf{DECaLS AUC} \\")
    caption = ("Zero-shot transfer from SDSS to the DESI Legacy Imaging Surveys. "
               "Models are trained only on Galaxy Zoo~2 cutouts and evaluated, without "
               "any adaptation or recalibration, on the Galaxy10 DECaLS images whose "
               "labels answer the same Galaxy Zoo question. $\\Delta$ is the accuracy "
               "lost in the transfer.")
    return _wrap("\n".join(lines), caption, "tab:cross", "llccccc", header)


def table_xai(runs: pd.DataFrame) -> str:
    path = config.RESULTS / "xai_summary.csv"
    if not path.exists():
        return "% explanation results not available yet\n"
    xai = pd.read_csv(path)
    lines = []
    for _, row in xai.sort_values(["arch", "label_mode"]).iterrows():
        acc = runs[runs["run_id"] == row["run_id"]]["test_accuracy"]
        lines.append(" & ".join([
            PRETTY_ARCH.get(row["arch"], row["arch"]),
            row["label_mode"],
            {"gradcam": "Grad-CAM", "gradcam++": "Grad-CAM++",
             "rollout": "rollout"}.get(row["method"], row["method"]),
            f"{acc.iloc[0]:.3f}" if len(acc) else "--",
            f"{row['deletion_auc']:.3f}",
            f"{row['insertion_auc']:.3f}",
            f"{row['background_reliance']:.3f}",
            f"{row['background_reliance_low_agreement']:.3f}",
        ]) + r" \\")

    header = (r"\textbf{Model} & \textbf{Labels} & \textbf{Method} & "
              r"\textbf{Accuracy} & \textbf{Del.}$\downarrow$ & "
              r"\textbf{Ins.}$\uparrow$ & \textbf{Bkg.}$\downarrow$ & "
              r"\textbf{Bkg. (low agr.)}$\downarrow$ \\")
    caption = ("Are the explanations faithful? \\emph{Del.} and \\emph{Ins.} are the "
               "areas under the deletion and insertion curves; a map that identifies "
               "the pixels the network actually uses gives a low deletion and a high "
               "insertion score. \\emph{Bkg.} is the share of attribution mass falling "
               "outside the galaxy's own footprint, where nothing relevant can be, and "
               "the last column restricts it to the galaxies the volunteers disagreed "
               "about.")
    return _wrap("\n".join(lines), caption, "tab:xai", "lllccccc", header)


# --------------------------------------------------------------------------- #
# Macros
# --------------------------------------------------------------------------- #

def macros(runs: pd.DataFrame) -> str:
    """Every number the prose quotes, as a \newcommand."""
    out = [r"% Generated by src/tables.py -- do not edit by hand.",
           r"% Rerun `python -m src.tables --out <paper>` after any change to the runs.",
           ""]

    def cmd(name: str, value) -> None:
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    ref = reference(runs)
    meta = read_json(config.ARRAYS / "gz2_meta.json") if (config.ARRAYS / "gz2_meta.json").exists() else {}
    ceiling = read_json(config.RESULTS / "ceiling.json") if (config.RESULTS / "ceiling.json").exists() else {}
    summary = read_json(config.RESULTS / "summary.json") if (config.RESULTS / "summary.json").exists() else {}

    # dataset
    if meta:
        splits = meta.get("split_counts", {})
        cmd("nGalaxies", f"{meta['n_galaxies']:,}".replace(",", "{,}"))
        for key in ("train", "val", "test"):
            if key in splits:
                cmd(f"n{key.capitalize()}", f"{splits[key]:,}".replace(",", "{,}"))
        cmd("featuredFraction", f"{100 * meta['featured_fraction']:.1f}")
        cmd("medianVotes", f"{meta['median_votes']:.0f}")
        cmd("minVotes", str(meta.get("min_votes", config.MIN_VOTES)))
        cmd("cacheSize", str(meta.get("cache_size", config.CACHE_SIZE)))

    # the vote-model ceiling
    if ceiling:
        raw = ceiling["raw"]
        cmd("voteCeiling", f"{100 * raw['bayes_accuracy']:.1f}")
        cmd("panelAgreement", f"{100 * raw['panel_agreement']:.1f}")
        cmd("labelNoiseRate", f"{100 * raw['label_noise_rate']:.1f}")
        deb = ceiling.get("debiased_sensitivity", {})
        if deb:
            cmd("voteCeilingDebiased", f"{100 * deb['bayes_accuracy']:.1f}")

    # best models
    for mode, tag in (("hard", "Hard"), ("soft", "Soft")):
        rows = ref[ref["label_mode"] == mode]
        if rows.empty:
            continue
        per_arch = rows.groupby("arch")["test_accuracy"].mean().sort_values()
        arch = per_arch.index[-1]
        best = rows[rows["arch"] == arch]
        cmd(f"best{tag}Arch", PRETTY_ARCH.get(arch, arch))
        cmd(f"best{tag}Accuracy", f"{100 * best['test_accuracy'].mean():.1f}")
        cmd(f"best{tag}AccuracySd", f"{100 * best['test_accuracy'].std(ddof=1):.1f}"
            if len(best) > 1 else "0.0")
        cmd(f"best{tag}Auc", f"{best['test_auc_roc'].mean():.3f}")
        cmd(f"best{tag}Ece", f"{best['test_ece'].mean():.3f}")

    # what soft labels buy, averaged over architectures
    pair = ref.pivot_table(index="arch", columns="label_mode",
                           values=["test_accuracy", "test_ece", "test_auc_roc"])
    if ("test_ece", "soft") in pair and ("test_ece", "hard") in pair:
        cmd("softEceGain", f"{100 * (pair[('test_ece', 'hard')] - pair[('test_ece', 'soft')]).mean():.1f}")
        cmd("softAccGain", f"{100 * (pair[('test_accuracy', 'soft')] - pair[('test_accuracy', 'hard')]).mean():.2f}")
        cmd("softAucGain", f"{1000 * (pair[('test_auc_roc', 'soft')] - pair[('test_auc_roc', 'hard')]).mean():.1f}")

    # accuracy in the extreme agreement bins, with the within-bin majority baseline
    # so the two ends are comparable (the class prior differs a lot between them)
    apath = config.RESULTS / "agreement.csv"
    if apath.exists():
        agr = pd.read_csv(apath)
        agr = agr[(agr["label_mode"] == "soft") & (agr["policy"] == "d4")]
        if not agr.empty:
            hi = agr[agr["bin"] == agr["bin"].max()]
            lo = agr[agr["bin"] == 0]
            cmd("accHighAgreement", f"{100 * hi['accuracy'].mean():.1f}")
            cmd("accLowAgreement", f"{100 * lo['accuracy'].mean():.1f}")
            cmd("shareLowAgreement", f"{100 * lo['share'].mean():.1f}")
            cmd("shareHighAgreement", f"{100 * hi['share'].mean():.1f}")
            for tag, frame in (("High", hi), ("Low", lo)):
                if "majority_baseline" in frame:
                    cmd(f"baseline{tag}Agreement",
                        f"{100 * frame['majority_baseline'].mean():.1f}")
                if "balanced_accuracy" in frame:
                    cmd(f"balAcc{tag}Agreement",
                        f"{100 * frame['balanced_accuracy'].mean():.1f}")
                if "lift" in frame:
                    cmd(f"lift{tag}Agreement", f"{100 * frame['lift'].mean():+.1f}")
                if "error_share" in frame:
                    cmd(f"errorShare{tag}Agreement",
                        f"{100 * frame['error_share'].mean():.0f}")
            # the three contested bins together, which is how the prose reads it
            contested = agr[agr["bin"] <= 2]
            if not contested.empty and "error_share" in contested:
                per_run = contested.groupby("run_id")[["share", "error_share"]].sum()
                cmd("shareContested", f"{100 * per_run['share'].mean():.0f}")
                cmd("errorShareContested", f"{100 * per_run['error_share'].mean():.0f}")

    # selective prediction
    spath = config.RESULTS / "selective.csv"
    if spath.exists():
        sel = pd.read_csv(spath)
        sel = sel.merge(runs[["run_id", "test_accuracy"]], on="run_id", how="left")
        soft = sel[sel["label_mode"] == "soft"]
        if not soft.empty:
            row = soft.loc[soft["coverage_at_99"].idxmax()]
            cmd("coverageAtNinetyNine", f"{100 * row['coverage_at_99']:.0f}")
            cmd("coverageAtNinetyEight", f"{100 * soft['coverage_at_98'].max():.0f}")
            cmd("coverageAtNinetyFive", f"{100 * soft['coverage_at_95'].max():.0f}")
            cmd("coverageArch", PRETTY_ARCH.get(row["arch"], str(row["arch"])))

    # orientation
    orient = runs[(runs["label_mode"] == "soft") & (runs["train_size"] == 0)
                  & (runs["finetune"] == "full") & (runs["loss"] == "bce")]
    plain = orient[(orient["policy"] == "none") & (~orient["orientation_pooled"])]
    d4 = orient[(orient["policy"] == "d4") & (~orient["orientation_pooled"])]
    if not plain.empty and not d4.empty:
        shared = set(plain["arch"]) & set(d4["arch"])
        gain = (d4[d4["arch"].isin(shared)].groupby("arch")["test_accuracy"].mean()
                - plain[plain["arch"].isin(shared)].groupby("arch")["test_accuracy"].mean())
        cmd("dihedralGain", f"{100 * gain.mean():.1f}")
        cmd("invarianceErrorPlain", f"{plain['d4_invariance_error'].mean():.3f}")
    pooled = orient[orient["orientation_pooled"]]
    if not pooled.empty and not d4.empty:
        shared = set(pooled["arch"]) & set(d4["arch"])
        diff = (pooled[pooled["arch"].isin(shared)].groupby("arch")["test_accuracy"].mean()
                - d4[d4["arch"].isin(shared)].groupby("arch")["test_accuracy"].mean())
        cmd("pooledGain", f"{100 * diff.mean():+.2f}")

    # ImageNet initialisation, done properly this time
    pre = runs[(runs["label_mode"] == "soft") & (runs["policy"] == "d4")
               & (runs["finetune"] == "full") & (runs["loss"] == "bce")
               & (runs["train_size"] == 0) & (~runs["orientation_pooled"])]
    if "pretrained" in pre and pre["pretrained"].nunique() > 1:
        yes = pre[pre["pretrained"]].groupby("arch")["test_accuracy"].mean()
        no = pre[~pre["pretrained"]].groupby("arch")["test_accuracy"].mean()
        shared = yes.index.intersection(no.index)
        if len(shared):
            cmd("pretrainGain", f"{100 * (yes[shared] - no[shared]).mean():+.1f}")
            cmd("pretrainGainMax", f"{100 * (yes[shared] - no[shared]).max():+.1f}")
    depth = runs[(runs["label_mode"] == "soft") & (runs["policy"] == "d4")
                 & (runs["loss"] == "bce") & (runs["train_size"] == 0)
                 & (~runs["orientation_pooled"]) & (runs["pretrained"])]
    full = depth[depth["finetune"] == "full"].groupby("arch")["test_accuracy"].mean()
    fz = depth[depth["finetune"] == "frozen"].groupby("arch")["test_accuracy"].mean()
    shared = full.index.intersection(fz.index)
    if len(shared):
        cmd("frozenPenalty", f"{100 * (full[shared] - fz[shared]).mean():.1f}")

    # cross-survey
    cpath = config.RESULTS / "cross_survey.csv"
    if cpath.exists():
        cross = pd.read_csv(cpath)
        if not cross.empty:
            cmd("decalsDropMean", f"{100 * cross['accuracy_drop'].mean():.1f}")
            best = cross.loc[cross["decals_accuracy"].idxmax()]
            cmd("decalsBestArch", PRETTY_ARCH.get(best["arch"], str(best["arch"])))
            cmd("decalsBestAccuracy", f"{100 * best['decals_accuracy']:.1f}")
            soft = cross[cross["label_mode"] == "soft"]["accuracy_drop"]
            hard = cross[cross["label_mode"] == "hard"]["accuracy_drop"]
            if len(soft) and len(hard):
                cmd("decalsDropSoft", f"{100 * soft.mean():.1f}")
                cmd("decalsDropHard", f"{100 * hard.mean():.1f}")

    # explanations
    xpath = config.RESULTS / "xai_summary.csv"
    if xpath.exists():
        xai = pd.read_csv(xpath)
        if not xai.empty:
            cmd("bkgRelianceMean", f"{100 * xai['background_reliance'].mean():.0f}")
            cmd("bkgRelianceLow", f"{100 * xai['background_reliance_low_agreement'].mean():.0f}")
            cmd("bkgRelianceHigh", f"{100 * xai['background_reliance_high_agreement'].mean():.0f}")
            cnn = xai[xai["method"].astype(str).str.startswith("gradcam")]
            vit = xai[xai["method"] == "rollout"]
            if not cnn.empty:
                cmd("bkgRelianceCNN", f"{100 * cnn['background_reliance'].mean():.0f}")
            if not vit.empty:
                cmd("bkgRelianceViT", f"{100 * vit['background_reliance'].mean():.0f}")

    # statistics
    fpath = config.RESULTS / "friedman.json"
    if fpath.exists():
        fried = read_json(fpath).get("soft_accuracy", {})
        if "friedman_p" in fried:
            p = fried["friedman_p"]
            cmd("friedmanP", "<0.001" if p < 1e-3 else f"{p:.3f}")
            cmd("friedmanArchitectures", str(fried["n_architectures"]))
            cmd("friedmanSeeds", str(fried["n_seeds"]))
            cmd("nemenyiCD", f"{fried['critical_difference']:.2f}")
    wpath = config.RESULTS / "wilcoxon.csv"
    if wpath.exists():
        wil = pd.read_csv(wpath)
        row = wil[(wil["knob"] == "label_mode") & (wil["level_a"] == "soft")
                  & (wil["level_b"] == "hard") & (wil["metric"] == "test_ece")]
        if not row.empty:
            p = float(row["p_value"].iloc[0])
            cmd("softVsHardEceP", "<0.001" if p < 1e-3 else f"{p:.3f}")
            cmd("softVsHardPairs", str(int(row["n_pairs"].iloc[0])))

    cmd("nRuns", str(int(summary.get("n_runs", len(runs)))))
    cmd("nArchitectures", str(int(runs["arch"].nunique())))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #

MACRO_NAMES = (
    "nGalaxies nTrain nVal nTest featuredFraction medianVotes minVotes cacheSize "
    "voteCeiling panelAgreement labelNoiseRate voteCeilingDebiased "
    "bestHardArch bestHardAccuracy bestHardAccuracySd bestHardAuc bestHardEce "
    "bestSoftArch bestSoftAccuracy bestSoftAccuracySd bestSoftAuc bestSoftEce "
    "softEceGain softAccGain softAucGain "
    "accHighAgreement accLowAgreement shareLowAgreement shareHighAgreement "
    "baselineHighAgreement baselineLowAgreement balAccHighAgreement "
    "balAccLowAgreement liftHighAgreement liftLowAgreement "
    "errorShareHighAgreement errorShareLowAgreement "
    "shareContested errorShareContested "
    "coverageAtNinetyNine coverageAtNinetyEight coverageAtNinetyFive coverageArch "
    "dihedralGain invarianceErrorPlain pooledGain "
    "pretrainGain pretrainGainMax frozenPenalty "
    "decalsDropMean decalsBestArch decalsBestAccuracy decalsDropSoft decalsDropHard "
    "bkgRelianceMean bkgRelianceLow bkgRelianceHigh bkgRelianceCNN bkgRelianceViT "
    "friedmanP friedmanArchitectures friedmanSeeds nemenyiCD "
    "softVsHardEceP softVsHardPairs nRuns nArchitectures"
).split()

PLACEHOLDER = r"\textcolor{red}{??}"

PLACEHOLDER_TABLE = "\n".join([
    r"\begin{table}[!htb]", r"\centering",
    r"\caption{Placeholder. Regenerate with \texttt{python -m src.tables --out <paper>} "
    r"once the runs have finished.}",
    r"\label{tab:%s}", r"\footnotesize",
    r"\begin{tabular}{l}", r"\toprule",
    r"\textbf{results pending} \\", r"\midrule",
    r"\textcolor{red}{??} \\", r"\bottomrule",
    r"\end{tabular}", r"\end{table}", "",
])


def write_placeholders(out_dir: Path) -> None:
    """Emit the same file set with every value marked in red.

    The manuscript can then be written, compiled and reviewed before the sweep has
    finished, and an unfilled number is impossible to miss in the PDF rather than
    quietly reading as a result.
    """
    for name, label in (("main", "main"), ("labels", "labels"),
                        ("orientation", "orientation"), ("protocol", "protocol"),
                        ("cross_survey", "cross"), ("xai", "xai")):
        (out_dir / f"{name}.tex").write_text(PLACEHOLDER_TABLE % label, encoding="utf-8")
    body = [r"% Placeholders. Regenerate with `python -m src.tables --out <paper>`.",
            r"% Every macro below resolves to a red ?? until the runs exist.", ""]
    body += [f"\\newcommand{{\\{name}}}{{{PLACEHOLDER}}}" for name in MACRO_NAMES]
    (out_dir / "numbers.tex").write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"wrote placeholder tables and {len(MACRO_NAMES)} macros to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=config.WORK / "paper_assets")
    ap.add_argument("--placeholders", action="store_true",
                    help="write the file set with every number marked, without "
                         "needing any results")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out / "tables")
    if args.placeholders:
        write_placeholders(out_dir)
        return

    runs = _load()

    selective = pd.read_csv(config.RESULTS / "selective.csv") \
        if (config.RESULTS / "selective.csv").exists() else None
    tracking = pd.read_csv(config.RESULTS / "vote_tracking.csv") \
        if (config.RESULTS / "vote_tracking.csv").exists() else None

    written = {
        "main.tex": table_main(runs, selective),
        "labels.tex": table_labels(runs, tracking),
        "orientation.tex": table_orientation(runs),
        "protocol.tex": table_protocol(runs),
        "cross_survey.tex": table_cross_survey(runs),
        "xai.tex": table_xai(runs),
        "numbers.tex": macros(runs),
    }
    for name, body in written.items():
        (out_dir / name).write_text(body, encoding="utf-8")
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
