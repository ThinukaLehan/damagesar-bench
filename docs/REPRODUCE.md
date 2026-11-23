# Reproducing the paper

Every development-phase number in the paper (`paper/`), except those of the fine-tuned reference (b01, whose GPU
training is not bit-exact), is listed in **`paper/expected_values.yaml`** with the run that produces it and how
closely a rerun must match. One notebook rebuilds them from the public data:

**`notebooks/reproduce/r01_reproduce_paper.ipynb`** (Colab, T4 GPU, about 4.5–5 hours)

Status (23 Nov 2026): its last complete run (22 Nov 2026) passed all 172 checks it then contained. The ViT-L
encoders, texture reference, Gate 1 and confound controls were added to it afterwards; a full rerun is still to
be done, and the paper says so.

1. Start a **fresh** session (Runtime → Disconnect and delete runtime), then Runtime → Change runtime type → T4 GPU.
2. Colab Secrets: `GH_TOKEN` (read access to this repository while it is private) and `HF_TOKEN`
   (Hugging Face read token; DINOv3 is gated: accept its licence on the model page first).
3. Run all. The last cells print a table with, for each paper number, the expected value, the value obtained
   and PASS / FAIL / MISSING, then a one-line verdict (`REPRODUCED: n of n checked values match`).

What it does, from scratch (nothing is read from earlier runs):

| Step | Paper numbers | Code |
|---|---|---|
| Download BRIGHT post-event SAR + label rasters (Hugging Face `Kullervo/BRIGHT`) and the instance labels (Zenodo 19216381) | — | notebook |
| Validate every tile | 4,246 tiles, 20 no-data exclusions | `bench validate` · `data/validate.py` |
| Build the building table | 244,976 buildings, 242,842 kept, exclusions, positives per event, prevalence range, Table II | `data/buildings.py` |
| Extract DINOv3 ViT-B/16 features (native, 0.3 m; outline, dilated 3 m) | — | `encoders/`, `pooling/`, `experiments/resolution.py` |
| Benchmark | Tables III and IV | `run.py`, `probe/`, `stats/` |
| Footprint-to-SAR offsets per tile (CPU, seed 0) | Section VI-A offsets table | `c2_offsets.py` |
| Controls + ImageNet baseline: DINOv3 ring 10 m, ImageNet outline / dilated 3 m / ring 10 m, image-free control | Section VI-D | `controls/`, `pooling/footprint.py`, `run.py` |
| Compare | all of the above | `reproduce.py` |

## Tolerances

* Counts, verdicts and convergence: exact.
* Building size in tokens (Table II): ±0.05 percentage points (the paper rounds to 0.1).
* AUROC: ±0.005. Encoders run in fp16 on a GPU, which is not bit-exact between runs or GPU types; probes,
  folds and the bootstrap (seed 0) are deterministic given the features.
* Comparison estimates ±0.005, interval bounds ±0.01.
* Offsets: medians ±0.05 m (tokens ±0.01), shares and direction consistency ±0.002.

## Recorded environment

Each run writes `environment.json` (Python, NumPy, pandas, SciPy, scikit-learn, PyArrow, rasterio, PyTorch,
Transformers, GPU, git commit, and the Hugging Face commit of the DINOv3 weights actually loaded). The
benchmark's `run_manifest.json` records the same environment plus a SHA-256 of every input file.
To load a specific weights revision, set `DSB_REVISION_DINOV3_VITB16=<commit>` before creating the encoder.

## The locked-test numbers

Numbers that use confirmatory labels are not in `expected_values.yaml` and are not rebuilt by r01. They come from
the single locked run of `notebooks/locked/k01_locked_confirmatory_test.ipynb` (22 Nov 2026, UTC, at the frozen
commit `92fba68`; see `EXPERIMENT_LOG.md`). `paper/tools/fill_locked.py` reads that run folder, re-derives the
RQ1 interval from the per-event gains and the verdict from the frozen decision table, and only then writes the
values used in the paper (`paper/locked_results/`).

## Data licences

BRIGHT SAR imagery is CC BY 4.0 (Capella Space, Umbra); optical imagery and labels are CC BY-NC. This repository
contains **code only**: the notebooks download the data from its original sources and never redistribute it.
