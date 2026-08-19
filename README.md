# galaxy-morphology

Code for a study of automated galaxy morphology classification that treats the
volunteer labels as what they are: the majority verdict of a panel of people who
often disagreed with each other.

The task is the binary one from the Galaxy Zoo decision tree — is the galaxy smooth
and rounded, or does it have features or a disk? — and the question we ask is not
only which network is most accurate but where the remaining error actually lives.
Twelve backbones are compared (three plain convolutional nets trained from scratch,
five ImageNet-pretrained convnets, four transformers), each trained twice: once on
the thresholded label everybody uses, and once on the vote fraction itself.

Concretely, the pipeline produces

* accuracy resolved by how much the volunteers agreed, against a ceiling computed
  from the vote statistics alone;
* calibration and selective-prediction curves, so the accuracy a survey pipeline
  needs can be traded against how much of the sky it hands to a human;
* an orientation study: augmentation over the dihedral group $D_4$ against
  invariance built into the network by construction;
* saliency maps scored rather than admired: deletion, insertion, and how much
  attribution mass lands outside the galaxy, quoted against the uniform-map null;
* zero-shot transfer from SDSS to the DESI Legacy Imaging Surveys.

## Data

Two public sources, neither redistributed here.

| | source | licence |
|---|---|---|
| Galaxy Zoo 2 catalogue | [`gz2_hart16.csv.gz`](https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz), Hart et al. (2016), via [data.galaxyzoo.org](https://data.galaxyzoo.org) | free to use with attribution |
| Galaxy Zoo 2 images | [Zenodo 3565489](https://zenodo.org/records/3565489), 243,434 SDSS cutouts, 3.4 GB | CC BY 4.0 |
| Galaxy10 DECaLS | [Zenodo 10845026](https://zenodo.org/records/10845026), 17,736 DESI Legacy cutouts | see record |

`src/download_data.py` fetches all three. Run it on a machine with network access:
the archive is large and the compute nodes have no outbound route.

Two things about how the labels are built are worth reading before using the
numbers.

**The label comes from task 01, not from `gz2_class`.** It would be easier to split
on the first letter of the `gz2_class` string, and that is what a lot of work on
this dataset does, including our own earlier attempt. It is the wrong split: the
early-type bin quietly absorbs lenticulars and edge-on disks. Task 01 of the
decision tree asks exactly the binary question we want to model, so we take its two
galaxy answers, renormalise them, and keep the resulting fraction $p$ as a
continuous target. Objects the volunteers mostly called stars or artefacts are
dropped, as are the few with fewer than 20 votes.

**And it comes from the raw fractions, not the debiased ones.** Hart et al. publish
both, and the debiased values are the better estimate of intrinsic morphology --
they correct for the fact that features get harder to see at higher redshift. That
correction is exactly why we do not train on them: it is a function of redshift,
which a cutout does not show, so a network asked to predict a debiased fraction is
being asked for something its input cannot determine. The vote model needs the same
thing for a different reason: it requires the label to be a threshold on a
proportion of a counted number of draws, which a debiased value is not. The
consequence is not cosmetic. On this catalogue the two thresholds disagree on about
a third of the galaxies and the featured fraction moves from roughly a quarter to
roughly three fifths, so the two definitions are tasks with very different
majority baselines. Because the choice matters that much, it is measured rather
than asserted: `hard_debiased` and `soft_debiased` train on the debiased fractions
while still being scored against the raw-label test set.

**Agreement is a first-class quantity.** From $p$ we keep $|2p-1|$, which is 0 when
the volunteers split evenly and 1 when they were unanimous. The splits are
stratified on it as well as on the class, so the agreement-resolved analysis is
comparable across seeds.

## Setup

```bash
git clone https://github.com/ralorin/galaxy-morphology.git
cd galaxy-morphology
bash setup_env.sh          # conda env `galaxy`, PyTorch cu121, then warms the weight cache
```

Point the code at two directories and keep them out of `$HOME` if quota is tight:

```bash
echo 'export GZM_DATA=$HOME/galaxy-morphology/data' >> ~/.bashrc
echo 'export GZM_WORK=$HOME/galaxy-morphology/work' >> ~/.bashrc
source ~/.bashrc
python config.py           # prints the resolved paths and what is already in place
```

The image cache is about 18 GB and the run directories a few GB more, so `$GZM_WORK`
wants to be on scratch.

## Running it

Steps 1 and 2 are login-node work. Everything else goes through `sbatch` from the
repository root, so that the logs land in `logs/`.

```bash
# 1. downloads (needs network; ~6 GB in total)
conda activate galaxy
python -m src.download_data

# 2. the job table, which also prints the array range to use below
python -m src.build_jobs
python -m src.build_jobs --list          # group sizes without writing anything

# 3. decode the images and build the label table and splits  (cpu, ~1 h)
sbatch slurm/01_prepare.sh

# 4. the training sweep  (gpu-small, one GPU per task)
sbatch --array=0-280%2 slurm/02_train_array.sh
#    or, to hold four GPUs and keep them all busy in one job:
#    FIRST=0 LAST=280 sbatch slurm/02b_train_multi.sh

# 5. cross-survey evaluation and the explanations  (gpu-small)
sbatch slurm/03_cross_survey.sh
sbatch slurm/04_xai.sh
#    --gallery-run <run_id> also keeps the cutouts and maps of one model as arrays,
#    which is what the manuscript's attribution figure is drawn from

# 6. analysis, statistics, figures and LaTeX tables  (cpu)
PAPER_DIR=$HOME/paper5 sbatch slurm/05_analysis.sh
#    STAGES=assets skips the two expensive stages and only redoes the figures,
#    the tables and the copy into results/, which takes minutes
```

Two repair paths worth knowing about, because both cost seconds and the alternative
is hours. `python -m src.xai --rebuild-summary` reassembles `xai_summary.csv` from the
per-run json files, and `python -m src.analysis --dataset-sample-only` writes the
sample the dataset figure needs without redoing the analysis.

Replace `280` with whatever `build_jobs` reports. The `%2` is the two-GPU-per-user
limit on `gpu-small`; raise it if the limit changes. Every training run checks for
its own `metrics.json` first and exits immediately if it is there, so a job that
hits the wall clock can be resubmitted over the same range without redoing work.

A single configuration, for a quick check that the environment is sound:

```bash
python -m src.train --arch resnet50 --label-mode soft --policy d4 --seed 0 --epochs 2
```

## Cost

Measured on one H100 at roughly 3,000 inference-equivalent images per second, over
the full training split and with early stopping typically landing around twelve
epochs:

| | per run | runs | GPU-h |
|---|---|---|---|
| ResNet-50, ViT-S/16, DeiT3-S, Swin-T | ~40 min | | |
| ConvNeXt-T, DenseNet-121, EfficientNetV2-S | ~45 min | | |
| ViT-B/16 | ~2.6 h | 10 | 26 |
| scratch CNNs at 128 px | ~10 min | | |
| orientation-pooled (eight passes per step) | ~5.3 h | 9 | 48 |

The design as shipped is 281 runs and about 200 GPU-hours: four days of wall clock
on the two GPUs `gpu-small` allows, or two days on four GPUs through
`02b_train_multi.sh`. The second is the better use of the machine here, and it is a
coherent request for `gpu-large` because all four cards stay busy for the whole
allocation.

`--groups` lets you run it in pieces. `main` (102 runs, 67 GPU-h) alone is enough
for the headline table, the agreement-resolved figure and the Friedman test; the
orientation group is the one to postpone if queue time is short, since it is a
quarter of the budget for a single ablation.

## Layout

```
config.py             paths, image geometry, label definition
src/
  registry.py         what each model is (no torch, so the analysis half stays light)
  experiment.py       what a run is: defaults, resolution, run id
  download_data.py    fetch the public datasets
  prepare_gz2.py      label table, splits, uint8 image cache
  prepare_decals.py   the same for Galaxy10 DECaLS
  datasets.py         Dataset and the four augmentation policies
  models.py           backbones, the scratch baselines, D4 pooling wrapper
  train.py            one run end to end; also cross-survey scoring
  build_jobs.py       the experiment design, expanded into jobs.csv
  analysis.py         aggregation, the vote-model ceiling, agreement-resolved tables
  stats.py            bootstrap CIs, McNemar, Friedman/Nemenyi, paired tests
  xai.py              Grad-CAM, attention rollout, faithfulness scores
  figures.py          every figure in the paper
  tables.py           every LaTeX table, plus numbers.tex
slurm/                batch scripts for the cluster partitions
```

## Output

Per run, in `$GZM_WORK/runs/<run_id>/`: `metrics.json` with the resolved
configuration and every metric, `history.csv` with the epoch trace,
`predictions_{val,test}.csv` with one probability per galaxy, and `checkpoint.pt`
for the runs the explanation and transfer stages need.

Aggregated, in `$GZM_WORK/results/`: `runs.csv` (one tidy row per run),
`agreement.csv`, `ceiling.json`, `selective.csv`, `calibration.csv`,
`cross_survey.csv`, `bootstrap.csv`, `mcnemar.csv`, `architecture_pairwise.csv`,
`friedman.json`, `wilcoxon.csv`, `xai_summary.csv`, `risk_coverage.csv`, and two
small arrays, `xai_gallery.npz` and `dataset_sample.npz`, which carry the cutouts and
attribution maps the two image figures need. With those in place every figure in the
paper redraws from a clone of this repository, with no cluster and no GPU.

`src/tables.py` also writes `numbers.tex`, a file of `\newcommand` macros for every
figure the manuscript quotes in prose. Nothing numeric is typed into the paper by
hand, so the text cannot drift away from the runs.

## Notes

- Reproducibility: seeds are fixed and the augmentation generator derives from them,
  but cuDNN is left in its non-deterministic mode for the training runs because it
  costs a fifth of the throughput and we average over five seeds anyway. Exact
  numbers will still shift slightly with the GPU model and the library versions.
- Swin-T only runs at 224 px; its window size fixes the feature-map geometry. It is
  therefore absent from the resolution sweep, and Grad-CAM is used for it instead of
  attention rollout, whose definition is unclear under shifted windows.
- The frozen fine-tuning mode is a linear probe and is reported as such. It is not
  the same experiment as random initialisation, and conflating the two is how a
  study can conclude that ImageNet pretraining does not help when what it actually
  measured was a head trained on random features. Both are in the design, separately.
- `.gitattributes` forces LF on `.sh` and `.py`, so `sbatch` will not trip over
  Windows line endings.
