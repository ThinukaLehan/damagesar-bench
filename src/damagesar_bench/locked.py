"""The locked confirmatory analysis (RQ1-RQ4), as frozen in docs/preregistration.md.

One function runs everything the locked-test notebook (k01) reports from the extracted features, so the same code
can be rehearsed on development events (k00) and tested on synthetic data before the freeze:

* RQ1 (primary): the DINOv3 model named in the freeze record vs ImageNet ViT-B/16, outline, native resolution.
* RQ2 (primary, only if Gate 1 passed): C2 matched vs outline pooling, DINOv3 ViT-B/16.
* Exploratory: DINOv3 ViT-B/16 vs ImageNet if RQ1 uses another DINOv3 model; RQ4 controls (image-free, 10 m ring).
* Pre-registered supporting evidence for RQ1 (``stats/supporting.py``): per-event paired tile-bootstrap
  differences and the one-sided sign-flip p-value.
* RQ3 (exploratory): C3 label-free prevalence on each confirmatory event (rule as frozen).

Comparisons use the confirmatory events only, at the level in the freeze record. Only RQ1 and RQ2 can be labelled
confirmatory in the report (``ComparisonSpec.primary``).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from .config import EventsConfig
from .recal.prevalence import estimate_all
from .run import ComparisonSpec, run_benchmark
from .stats.supporting import per_event_paired, summarise

IMAGENET = "imagenet_vitb16@native"
VITB16 = "dinov3_vitb16@native"


def locked_comparisons(rq1_encoder: str, run_rq2: bool) -> list[ComparisonSpec]:
    d = f"{rq1_encoder}@native"
    comps = [ComparisonSpec((d, "outline"), (IMAGENET, "outline"), "RQ1 DINOv3 vs ImageNet (outline)")]
    if run_rq2:
        comps.append(ComparisonSpec((VITB16, "matched"), (VITB16, "outline"), "RQ2 C2 matched vs outline (DINOv3)"))
    if d != VITB16:
        comps.append(ComparisonSpec((VITB16, "outline"), (IMAGENET, "outline"),
                                    "exploratory DINOv3 ViT-B vs ImageNet (outline)", primary=False))
    comps += [ComparisonSpec((d, "outline"), ("imagefree", "geometry"), "RQ4 DINOv3 vs image-free", primary=False),
              ComparisonSpec((d, "outline"), (d, "ring10m"), "RQ4 building vs surroundings (DINOv3)",
                             primary=False)]
    return comps


def run_locked_analysis(feature_files: list[str | Path], events: EventsConfig, out_dir: str | Path, *,
                        rq1_encoder: str, run_rq2: bool, level: float, n_boot: int = 2000, seed: int = 0,
                        today: date | None = None,
                        c3_source_roles: tuple[str, ...] = ("development", "secondary")) -> dict:
    """Run the locked comparisons, the supporting evidence and RQ3; write everything under ``out_dir``.

    ``c3_source_roles``: the roles whose held-out scores calibrate C3 (frozen rule: development + secondary).
    Only the k00 rehearsal changes it, because there the development events play the confirmatory role.
    """
    out_dir = Path(out_dir)
    out = run_benchmark(feature_files, locked_comparisons(rq1_encoder, run_rq2), events, out_dir / "bench_run",
                        n_boot=n_boot, seed=seed, level=level, event_set="confirmatory", only_compared=True,
                        today=today)
    comps = pd.read_csv(out / "comparisons.csv")
    scores = pd.read_parquet(out / "fold_scores.parquet")
    conf = events.keys_with_role("confirmatory")
    d = f"{rq1_encoder}@native"
    per_event = per_event_paired(scores, (d, "outline"), (IMAGENET, "outline"), conf, n_boot=n_boot, seed=seed)
    per_event.to_csv(out_dir / "rq1_per_event.csv", index=False)
    sup = summarise(per_event)
    (out_dir / "rq1_supporting.json").write_text(json.dumps(asdict(sup), indent=1), encoding="utf-8")
    c3 = estimate_all(scores, d, "outline", source_roles=c3_source_roles)
    c3 = c3[c3["event"].isin(conf)].reset_index(drop=True)
    c3.to_csv(out_dir / "c3_confirmatory.csv", index=False)
    return {"bench_run": out, "comparisons": comps, "per_event": per_event, "supporting": sup, "c3": c3}
