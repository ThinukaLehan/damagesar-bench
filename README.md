# DamageSAR-Bench

A reproducible, leakage-audited toolkit for **building-level damage classification from SAR on unseen disasters**.

Building footprints are given; the toolkit classifies each one as damaged or intact from post-event SAR, testing frozen image encoders on disasters they have never seen (leave-one-event-out on the [BRIGHT](https://github.com/ChenHongruixuan/BRIGHT) dataset). It adds two tested components: **projected footprints** (C2), which move outlines to where the radar sees each building, and **event-prior recalibration** (C3), which estimates a new event's damage rate without labels.

Final-year project, 6SENG010C Software Engineering Final Project (2026–27), Thinuka Lehan.

> **Decision support only.** Not validated for operational rescue decisions.

## Status

| Command | What it does | Status |
|---|---|---|
| `bench validate` | Checks data: pairing, shape, CRS, geotransform, label values, no-data | ✅ |
| `bench audit` | Writes `splits.json` (leave-one-event-out folds); leakage checks | ✅ splits · ⏳ leakage |
| `bench status` | Shows event roles and whether confirmatory labels are locked | ✅ |
| `bench buildings` | Builds `buildings.parquet` from BRIGHT instance labels (footprints, ground area, no-data rule, labels where unlocked) | ✅ |
| `bench extract` | Encodes tiles and pools building features (outline, dilated); resumable per tile | ✅ (projected pooling ⏳) |
| `bench run` | Probes, per-event metrics, comparisons with event-level verdicts | ✅ · recalibration, controls ⏳ |
| `bench report` | Markdown report from a run folder | ✅ v0 · HTML ⏳ |
| `bench new-event` | Damaged-building estimate for an unlabelled event, or "estimate unreliable" | ⏳ Stage 5 |

## Install

```bash
git clone https://github.com/<you>/damagesar-bench.git
cd damagesar-bench
pip install -e ".[dev]"      # core + tests
pip install -e ".[ml]"       # later: encoders, probes, statistics
```

On Colab or Kaggle, do **not** use `pip install -U`; install with `pip install -e .` only.

## Quick start (Colab)

```bash
# BRIGHT downloaded and unzipped to /content/bright/{post-event,target}
bench status
bench validate --root /content/bright --out /content/drive/MyDrive/DamageSAR-Bench/04_outputs/runs/<date>_validate
bench audit --out /content/drive/MyDrive/DamageSAR-Bench/01_data/derived
```

## Pre-registration guard

Event roles live in [`configs/events.yaml`](configs/events.yaml): 6 confirmatory, 5 development, 3 secondary events. Until the plan freeze (**10 Jan 2027**), the code refuses to read labels of confirmatory events, and validation reports only their schema, not class counts. Before the freeze, probes train on development and secondary events only. See [`docs/preregistration.md`](docs/preregistration.md).

## Layout

```
configs/            events.yaml (roles, freeze date) · bench.yaml (paths, no-data rule)
src/damagesar_bench/
  data/             discovery, validation, no-data rule, roles + guard, splits   (FR1–FR4)
  encoders/         frozen-encoder plugins                                        (FR5)
  pooling/          outline · dilated · projected footprints (C2)                 (FR6)
  probe/            per-fold logistic probes                                      (FR7)
  recal/            EM / BBSE recalibration + abstain rule (C3)                   (FR8)
  controls/         shortcut and leakage controls                                 (FR9)
  stats/            metrics, bootstrap, DeLong, sign-flip, t-interval, random effects (FR10)
  report/           HTML report + model card                                      (FR11)
  drive.py          Google Drive folder layout and per-run output folders
tests/              pytest suite (runs on CPU in CI)
notebooks/checks/   reproduction checks c01–c05 (Colab)
notebooks/          thin Colab runners
docs/               pre-registration, ADRs, Google Drive layout (drive_layout.md)
```

## Development

```bash
pytest --cov=damagesar_bench
```

Every change is logged in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md); AI assistance and how it was verified is logged in [`AI_USE_LOG.md`](AI_USE_LOG.md); design decisions are recorded as ADRs in [`docs/adr/`](docs/adr/).

## AI assistance

Parts of the code, tests and documentation were drafted with Claude (Anthropic), a generative-AI tool, and were reviewed, tested and verified by the author, who is responsible for all content. AI use was approved by the project supervisor. Every use, and how it was checked, is recorded in [`AI_USE_LOG.md`](AI_USE_LOG.md).

## Citation

If you use this toolkit, please cite it using [`CITATION.cff`](CITATION.cff).

## Data and licences

BRIGHT SAR imagery is CC BY 4.0 (Capella, Umbra). BRIGHT optical imagery and labels are mostly CC BY-NC 4.0, for academic use only. **No BRIGHT data is stored in or redistributed from this repository.** Code: MIT.
