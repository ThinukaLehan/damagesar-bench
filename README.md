# DamageSAR-Bench

A reproducible, leakage-audited toolkit for **building-level damage classification from SAR on unseen disasters**.

Building footprints are given; the toolkit classifies each one as damaged or intact from post-event SAR, testing frozen image encoders on disasters they have never seen (leave-one-event-out on the [BRIGHT](https://github.com/ChenHongruixuan/BRIGHT) dataset). It adds image-free and neighbourhood (ring) controls at building level, and two candidate methods tested under gates fixed in advance: **image-matched footprint pooling** (C2), which moves outlines to where the SAR signal is measured, and **label-free recalibration** (C3), which estimates a new event's damage rate without labels. Neither method passed its gate on the development events; both are reported as negative results.

Final-year project, 6SENG010C Software Engineering Final Project (2026–27), Thinuka Lehan. The paper based on it is by Thinuka Lehan, Achala Aponso and Maninda Edirisooriya (Informatics Institute of Technology).

## Project state (23 Nov 2026)

- **Analysis plan frozen on 22 Nov 2026** and deposited on Zenodo: [10.5281/zenodo.24443763](https://doi.org/10.5281/zenodo.24443763); git tag `prereg-freeze` (commit `92fba68`); freeze record in [`docs/FREEZE.md`](docs/FREEZE.md).
- **Development-phase experiments** (benchmark, controls, five encoders, Gate 1, recalibration, fine-tuned and texture references, confound controls) were run on the development events; every run and its result is recorded in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
- **Locked confirmatory test (k01)** was run once at the frozen commit on 22 Nov 2026 (UTC). A first run stopped for lack of memory before any result was written; it was rerun unchanged on a High-RAM runtime. Result for RQ1 (satellite-pretrained DINOv3 ViT-L minus ImageNet ViT-B, five confirmatory events): −0.001 AUROC, 97.5% interval [−0.147, +0.145], **inconclusive**. Details: [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) and [`paper/locked_results/`](paper/locked_results/), checked against the frozen decision table by `paper/tools/fill_locked.py`.
- **Paper** sources (IEEE JSTARS and IEEE Access versions) are in [`paper/`](paper/).

> **Decision support only.** Not validated for operational rescue decisions.

## Status

| Command | What it does | Status |
|---|---|---|
| `bench validate` | Checks data: pairing, shape, CRS, geotransform, label values, no-data | ✅ |
| `bench audit` | Writes `splits.json` (leave-one-event-out folds); the confirmatory-label lock itself is enforced at every hand-over (`contracts.py`) | ✅ |
| `bench status` | Shows event roles and whether confirmatory labels are locked | ✅ |
| `bench buildings` | Builds `buildings.parquet` from BRIGHT instance labels (footprints, ground area, no-data rule, labels where unlocked) | ✅ |
| `bench extract` | Encodes tiles and pools building features (outline, dilated, 10 m ring); resumable per tile. Image-matched pooling (C2) needs per-tile shifts and runs from Python (`c2_offsets.py`, g01 notebook) | ✅ |
| `bench run` | Probes, per-event metrics, comparisons with event-level verdicts; the image-free control runs as a features file | ✅ |
| Python API / notebooks | Label-free recalibration (C3, `recal/`), locked analysis (`locked.py`), references (`baselines/`), masked-building control (`controls/masked.py`) | ✅ (masked control built, not yet run) |
| `bench report` | Markdown report from a run folder | ✅ Markdown · ⏳ HTML report and model card |
| `bench new-event` | Damaged-building estimate for an unlabelled event, or "estimate unreliable" | ⏳ planned |

## Install

```bash
git clone https://github.com/ThinukaLehan/damagesar-bench.git
cd damagesar-bench
pip install -e ".[dev]"      # core + tests
pip install -e ".[ml]"       # encoders (PyTorch, Transformers)
```

On Colab or Kaggle, do **not** use `pip install -U`; install with `pip install -e .` only.

## Quick start (Colab)

```bash
# BRIGHT downloaded and unzipped to /content/bright/{post-event,target}
bench status
bench validate --root /content/bright --out /content/drive/MyDrive/DamageSAR-Bench/04_outputs/runs/<date>_validate
bench audit --out /content/drive/MyDrive/DamageSAR-Bench/01_data/derived
```

## Reproducing the paper

The development-phase numbers of the paper (all except those of the fine-tuned reference, whose GPU training is not bit-exact) are listed in `paper/expected_values.yaml`; `notebooks/reproduce/r01_reproduce_paper.ipynb` rebuilds them from the public data in a fresh Colab session and prints PASS/FAIL for each. Its last complete run (22 Nov 2026) passed all 172 checks it then contained; the ViT-L encoders, texture reference, Gate 1 and confound controls were added to it afterwards, and a full rerun is still to be done. The locked-test numbers come from the k01 run folder and are checked by `paper/tools/fill_locked.py`. See [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Pre-registration guard

Event roles live in [`configs/events.yaml`](configs/events.yaml): **5 confirmatory** (Haiti, Türkiye, Morocco, Noto, Bata), **5 development** (Beirut, La Palma, Hawaii, Congo, Libya) and **4 secondary** (Marshall, Mexico, Myanmar, Ukraine) events. Ukraine moved from confirmatory to secondary on 22 Nov 2026 under the plan's own rule (no instance labels, and no open footprint source passed the agreement test). From 9 Nov 2026 until the plan freeze on **22 Nov 2026**, the code refused to read labels of confirmatory events, and probes trained on development and secondary events only; the locked test (k01) refuses to start unless the freeze record, git tag and DOI are in place. The planned freeze date was 10 Jan 2027; it was brought forward with the supervisor's agreement. See [`docs/preregistration.md`](docs/preregistration.md) and [`docs/FREEZE.md`](docs/FREEZE.md).

## Layout

```
configs/            events.yaml (roles, freeze date) · bench.yaml (paths, no-data rule)
src/damagesar_bench/
  data/             discovery, validation, no-data rule, roles + guard, splits   (FR1–FR4)
  encoders/         frozen-encoder plugins                                        (FR5)
  pooling/          outline · dilated · 10 m ring · image-matched (C2)            (FR6)
  probe/            per-fold logistic probes                                      (FR7)
  recal/            EM / BBSE recalibration + abstain rule (C3)                   (FR8)
  controls/         image-free control · masked-building control                  (FR9)
  baselines/        fine-tuned ViT-B and texture + random-forest references
  stats/            metrics, bootstrap, DeLong, sign-flip, t-interval, Hartung-Knapp (FR10)
  report/           (planned) HTML report + model card                            (FR11)
  freeze.py         freeze check used by the locked test
  locked.py         the locked confirmatory analysis (k01), rehearsed in k00
  drive.py          Google Drive folder layout and per-run output folders
tests/              pytest suite (runs on CPU in CI)
notebooks/checks/   data checks (c01–c07) and the locked-analysis rehearsal (k00)
notebooks/pipeline/ experiment runners (p01–p06, g01, c301, b01, b02)
notebooks/locked/   the locked confirmatory test (k01)
notebooks/reproduce/ r01, rebuilds the paper's development-phase numbers
paper/              paper sources (JSTARS, IEEE Access), expected values, locked results, tools
docs/               pre-registration, freeze record, reproduction guide, ADRs, Google Drive layout
```

## Development

```bash
pytest --cov=damagesar_bench
```

Every change is logged in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md); AI assistance and how it was verified is logged in [`AI_USE_LOG.md`](AI_USE_LOG.md); design decisions are recorded as ADRs in [`docs/adr/`](docs/adr/).

## AI assistance

Parts of the code, tests, documentation and paper text were drafted with Claude (Anthropic), a generative-AI tool, and were reviewed, tested and verified by the author, who is responsible for all content. AI use was approved by the project supervisor. Every use, and how it was checked, is recorded in [`AI_USE_LOG.md`](AI_USE_LOG.md).

## Citation

If you use this toolkit, please cite it using [`CITATION.cff`](CITATION.cff).

## Data and licences

BRIGHT SAR imagery is CC BY 4.0 (Capella, Umbra). BRIGHT optical imagery and labels are mostly CC BY-NC 4.0, for academic use only. **No BRIGHT data is stored in or redistributed from this repository.** Code: MIT.
