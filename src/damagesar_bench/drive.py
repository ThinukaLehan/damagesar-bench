"""Google Drive folder layout for the project (FYP Google account).

Everything lives under one root folder, ``MyDrive/DamageSAR-Bench``. Notebooks never invent their
own paths: they ask this module where to read and write, so the Drive stays organised.

    DamageSAR-Bench/
      00_admin/                  briefings, declarations, exported docs
      01_data/
        raw/bright/              BRIGHT zips (optional cache; Colab downloads to /content)
        raw/instance_labels/     BRIGHT instance-label zips
        external/                open footprints, heights, SAR catalogue extracts
        derived/                 splits.json, buildings.parquet, c2_metadata_by_event.csv
      02_models/
        checkpoints/             third-party weights (e.g. BRIGHT UNet from Zenodo)
        features/                cached encoder features, one folder per encoder
      03_notebooks/
        pipeline/                main runners (extract, run, report)
        checks/                  test / reproduction notebooks (00-04)
        scratch/                 throwaway notebooks, safe to delete
      04_outputs/
        runs/<date>_<name>_<commit>/   one folder per run: CSVs, figures, executed notebook
        reports/                 bench report HTML
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

ROOT_NAME = "DamageSAR-Bench"
COLAB_DRIVE = Path("/content/drive/MyDrive")

FOLDERS = (
    "00_admin",
    "01_data/raw/bright",
    "01_data/raw/instance_labels",
    "01_data/external",
    "01_data/derived",
    "02_models/checkpoints",
    "02_models/features",
    "03_notebooks/pipeline",
    "03_notebooks/checks",
    "03_notebooks/scratch",
    "04_outputs/runs",
    "04_outputs/reports",
)

# Short names notebooks can use instead of paths.
ALIASES = {
    "admin": "00_admin",
    "raw_bright": "01_data/raw/bright",
    "raw_instance_labels": "01_data/raw/instance_labels",
    "external": "01_data/external",
    "derived": "01_data/derived",
    "checkpoints": "02_models/checkpoints",
    "features": "02_models/features",
    "nb_pipeline": "03_notebooks/pipeline",
    "nb_checks": "03_notebooks/checks",
    "nb_scratch": "03_notebooks/scratch",
    "runs": "04_outputs/runs",
    "reports": "04_outputs/reports",
}


def project_root(drive_root: str | Path = COLAB_DRIVE) -> Path:
    return Path(drive_root) / ROOT_NAME


def path(alias: str, drive_root: str | Path = COLAB_DRIVE) -> Path:
    """Folder for a short alias, e.g. ``path("derived")``."""
    if alias not in ALIASES:
        raise KeyError(f"unknown folder alias {alias!r}; choose from {sorted(ALIASES)}")
    return project_root(drive_root) / ALIASES[alias]


def ensure_layout(drive_root: str | Path = COLAB_DRIVE) -> list[Path]:
    """Create every project folder (idempotent). Returns the folders that exist afterwards."""
    root = project_root(drive_root)
    made = []
    for rel in FOLDERS:
        p = root / rel
        p.mkdir(parents=True, exist_ok=True)
        made.append(p)
    return made


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "run"


def run_dir(name: str, commit: str | None = None, today: date | None = None,
            drive_root: str | Path = COLAB_DRIVE, create: bool = True) -> Path:
    """Folder for one run: ``04_outputs/runs/2026-11-09_02-token-size-beirut_508730c``."""
    today = today or date.today()
    parts = [today.isoformat(), _slug(name)]
    if commit:
        parts.append(_slug(commit)[:7])
    p = path("runs", drive_root) / "_".join(parts)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p
