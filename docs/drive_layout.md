# Google Drive layout (FYP Google account)

All project files in Google Drive live under **one root folder**, `MyDrive/DamageSAR-Bench`, in the
FYP Google account (`thinukalehanfyp@gmail.com`). Notebooks never invent their own paths: they use
`src/damagesar_bench/drive.py` (or the identical setup cell in each check notebook), which creates the
folders below and gives every run its own folder.

```
DamageSAR-Bench/
├── 00_admin/                     briefings, declarations, exported docs (PDF)
├── 01_data/
│   ├── raw/
│   │   ├── bright/               BRIGHT zips (optional cache; Colab downloads to /content, faster)
│   │   └── instance_labels/      BRIGHT instance-label zips (cvprw26_trainval_instance_labels.zip)
│   ├── external/                 open footprints, building heights, SAR catalogue extracts
│   └── derived/                  splits.json, buildings.parquet, c2_metadata_by_event.csv,
│                                 bright_event_summary.csv (3 Nov)
├── 02_models/
│   ├── checkpoints/              third-party weights (BRIGHT UNet, Zenodo 15349462)
│   └── features/                 cached encoder features, one subfolder per encoder
├── 03_notebooks/
│   ├── pipeline/                 main runners: extract, run, report
│   ├── checks/                   test and reproduction notebooks (00–04), executed copies
│   └── scratch/                  throwaway notebooks, safe to delete
└── 04_outputs/
    ├── runs/                     one folder per run: <date>_<notebook>[_<commit>]/
    │                             CSVs, figures, executed notebook copy
    └── reports/                  bench report HTML
```

## Rules

1. **One run, one folder** in `04_outputs/runs/`, named `<YYYY-MM-DD>_<notebook-name>[_<commit>]`.
   Nothing is written straight into `04_outputs/`.
2. **Source notebooks live in GitHub**, not Drive. Colab opens them from the repo. After a run, the
   executed copy (with outputs) is saved to `03_notebooks/checks/` (or `pipeline/`) and also into that
   run's folder.
3. **Raw data is never edited.** Anything computed from it goes to `01_data/derived/`.
4. **Scratch is disposable.** Anything worth keeping moves out of `03_notebooks/scratch/`.
5. **Licences:** BRIGHT labels and optical images are CC BY-NC (academic use only). They may sit in this
   private Drive but are never shared or published from it.
6. **Every run is logged** as one line in `EXPERIMENT_LOG.md` (repo), with its run folder name.

## Moving earlier files

The 3 Nov outputs (`bright_event_summary.csv`, `splits.json`, `tile_nodata.csv`,
`B4_unet_zeroshot_dev.csv`) are in the old Google account under `MyDrive/damagesar/data/`. Copy them into
`01_data/derived/` here (via Drive "Share" or download and upload), so notebook 00 can compare against them.
