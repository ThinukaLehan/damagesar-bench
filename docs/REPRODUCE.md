# Reproducing the paper

Every number in the draft paper (`paper/`) is listed in **`paper/expected_values.yaml`** with the run that
produces it and how closely a rerun must match. One notebook rebuilds all of them from the public data:

**`notebooks/reproduce/r01_reproduce_paper.ipynb`** (Colab, T4 GPU, about 60–75 minutes)

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
| Compare | all of the above | `reproduce.py` |

## Tolerances

* Counts, verdicts and convergence: exact.
* Building size in tokens (Table II): ±0.05 percentage points (the paper rounds to 0.1).
* AUROC: ±0.005. Encoders run in fp16 on a GPU, which is not bit-exact between runs or GPU types; probes,
  folds and the bootstrap (seed 0) are deterministic given the features.
* Comparison estimates ±0.005, interval bounds ±0.01.

## Recorded environment

Each run writes `environment.json` (Python, NumPy, pandas, SciPy, scikit-learn, PyArrow, rasterio, PyTorch,
Transformers, GPU, git commit, and the Hugging Face commit of the DINOv3 weights actually loaded). The
benchmark's `run_manifest.json` records the same environment plus a SHA-256 of every input file.
To load a specific weights revision, set `DSB_REVISION_DINOV3_VITB16=<commit>` before creating the encoder.

## What is not checked before 10 Jan 2027

Numbers that need confirmatory labels (pixel class shares over all 14 events, class counts and prevalence of the
confirmatory events, all confirmatory results) are locked until the plan freeze and are not in the file; they
are added, and checked the same way, after the freeze. Footprint-to-SAR offsets are reproduced by
`notebooks/checks/c07_c2_offsets.ipynb` and will be added to the file once their numbers replace the
31 Oct audit's in the paper.

## Data licences

BRIGHT SAR imagery is CC BY 4.0 (Capella Space, Umbra); optical imagery and labels are CC BY-NC. This repository
contains **code only**: the notebooks download the data from its original sources and never redistribute it.
