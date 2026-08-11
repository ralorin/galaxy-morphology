# Results

Everything here is written by `python -m src.publish` after the analysis has run,
and it is the record that lets a number in the paper be checked without a GPU. The
large and regenerable things — image caches, checkpoints, per-galaxy prediction
files — stay in `$GZM_WORK` and are not committed.

## Aggregate tables

| file | contents |
|---|---|
| `runs.csv` | one row per training run: the resolved configuration and every metric, including the metrics at the validation-chosen threshold and after temperature scaling |
| `summary.json` | the handful of figures quoted in the abstract |
| `ceiling.json` | the vote-model quantities: best attainable accuracy against the recorded label, panel-to-panel agreement, label noise rate, and the debiased sensitivity check |
| `agreement.csv` | accuracy, balanced accuracy, the within-bin majority baseline, the lift over it, and the share of the model's total errors, per volunteer-agreement bin per run |
| `selective.csv` | risk–coverage summaries and the coverage reachable at 95, 98 and 99 per cent accuracy |
| `calibration.csv` | reliability curves for the reference runs |
| `vote_tracking.csv` | correlation between the predicted probability and the volunteer vote fraction |
| `cross_survey.csv` | zero-shot transfer from Galaxy Zoo 2 to Galaxy10 DECaLS |
| `bootstrap.csv` | percentile bootstrap confidence intervals over the test set |
| `mcnemar.csv` | paired exact McNemar tests for the comparisons the paper claims |
| `architecture_pairwise.csv` | Holm-corrected pairwise McNemar between architectures, on seed-averaged predictions |
| `friedman.json` | Friedman test across architectures with seeds as blocks, plus Nemenyi ranks |
| `wilcoxon.csv` | paired signed-rank tests for each ablation knob |
| `xai_summary.csv` | faithfulness and background-excess scores per explained run |

## Per-run and per-galaxy

`metrics/<run_id>.json` — one file per run, about four kilobytes each: the resolved
configuration, wall-clock times, parameter counts, the empirical $D_4$ invariance
error, and the full metric set on the test, validation and cross-survey splits.

`xai/<run_id>.csv` — one row per explained galaxy: deletion and insertion areas,
background reliance and its excess over the uniform null, the segmented footprint
fraction, the volunteer vote fraction and agreement, and whether the model was right.

## Figures and tables

`figures/` and `tables/` hold the exact assets the manuscript uses, so the version
of a figure that appears in the paper can be recovered from the same commit as the
numbers behind it.

## Reading a number

Every figure quoted in the manuscript prose is a LaTeX macro defined in
`tables/numbers.tex`, and every one of those is computed by `src/tables.py` from
`runs.csv` and the files above. Nothing is typed by hand, so a value in the paper
can be traced to a column here and from there to a `metrics/<run_id>.json`.
