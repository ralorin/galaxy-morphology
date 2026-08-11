"""Copy the aggregated results out of $GZM_WORK and into the repository.

    python -m src.publish
    python -m src.publish --paper /path/to/paper5     # also pick up its tables

$GZM_WORK lives on scratch and is not version controlled, which is right for the
things that are large or reproducible on demand: the image caches, the checkpoints,
and the per-galaxy prediction files. What belongs in the repository is the record
that lets someone check a number in the paper without a GPU, and that record is
small.

So this copies the aggregate tables, the per-run metrics, the explanation scores and
the figures into `results/`, where git can see them. It skips
`predictions_*.csv`: there are three of them per run at a few megabytes each, which
would add most of a gigabyte to the history to store something any reader can
regenerate from the checkpoints.

Run it on the login node after the analysis, then commit:

    python -m src.publish
    git add results && git commit -m "Add results of the full sweep" && git push
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import config
from src.common import ensure_dir

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "results"

# aggregate tables written by src.analysis and src.stats
AGGREGATES = (
    "runs.csv", "summary.json", "ceiling.json", "agreement.csv", "selective.csv",
    "calibration.csv", "vote_tracking.csv", "cross_survey.csv", "bootstrap.csv",
    "mcnemar.csv", "architecture_pairwise.csv", "friedman.json", "wilcoxon.csv",
    "xai_summary.csv",
)


def copy_aggregates() -> int:
    ensure_dir(TARGET)
    n = 0
    for name in AGGREGATES:
        src = config.RESULTS / name
        if src.exists():
            shutil.copy2(src, TARGET / name)
            n += 1
        else:
            print(f"  missing, skipped: {name}")
    # The dataset record lives with the arrays rather than the results, but a dozen of
    # the manuscript's numbers come from it, so without it the published results cannot
    # regenerate the tables on their own.
    meta = config.ARRAYS / "gz2_meta.json"
    if meta.exists():
        shutil.copy2(meta, TARGET / "gz2_meta.json")
        n += 1
    else:
        print("  missing, skipped: gz2_meta.json")
    return n


def copy_run_metrics() -> int:
    """One metrics.json per run: the resolved configuration and every metric.

    About four kilobytes each, so the whole sweep is a couple of megabytes and it
    is the file a reader would actually want in order to audit a table.
    """
    out = ensure_dir(TARGET / "metrics")
    n = 0
    for path in sorted(config.RUNS.glob("*/metrics.json")):
        shutil.copy2(path, out / f"{path.parent.name}.json")
        n += 1
    return n


def copy_xai() -> int:
    src_dir = config.RESULTS / "xai"
    if not src_dir.is_dir():
        return 0
    out = ensure_dir(TARGET / "xai")
    n = 0
    for path in sorted(src_dir.glob("*.csv")):
        shutil.copy2(path, out / path.name)
        n += 1
    return n


def copy_figures(paper: Path | None) -> int:
    out = ensure_dir(TARGET / "figures")
    n = 0
    for src_dir in filter(None, (config.FIGURES, paper / "figures" if paper else None)):
        if not Path(src_dir).is_dir():
            continue
        for path in sorted(Path(src_dir).glob("*.p*")):   # pdf and png
            shutil.copy2(path, out / path.name)
            n += 1
    return n


def copy_tables(paper: Path | None) -> int:
    if paper is None or not (paper / "tables").is_dir():
        return 0
    out = ensure_dir(TARGET / "tables")
    n = 0
    for path in sorted((paper / "tables").glob("*.tex")):
        shutil.copy2(path, out / path.name)
        n += 1
    return n


def directory_size(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", type=Path, default=None,
                    help="directory holding the manuscript's figures/ and tables/, "
                         "so that the exact assets it uses are archived too. "
                         "Defaults to ./paper5 if that exists.")
    args = ap.parse_args()

    # The manuscript is often kept beside the code for convenience. It is ignored by
    # git (see .gitignore), but its generated assets are worth archiving, so pick it
    # up automatically when it is there.
    if args.paper is None and (REPO / "paper5").is_dir():
        args.paper = REPO / "paper5"
        print(f"found the manuscript at {args.paper}")

    print(f"copying from {config.RESULTS}")
    counts = {
        "aggregate tables": copy_aggregates(),
        "per-run metrics": copy_run_metrics(),
        "explanation scores": copy_xai(),
        "figures": copy_figures(args.paper),
        "LaTeX tables": copy_tables(args.paper),
    }
    print()
    for label, n in counts.items():
        print(f"  {label:20s} {n:4d}")
    print(f"\n{TARGET} is now {directory_size(TARGET):.1f} MB")
    print("\ncommit it with:")
    print("  git add results && git commit -m \"Add results of the full sweep\" && git push")


if __name__ == "__main__":
    main()
