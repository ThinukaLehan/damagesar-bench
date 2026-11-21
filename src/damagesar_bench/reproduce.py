"""Check a fresh run against every number the paper reports (``paper/expected_values.yaml``).

``collect_actual`` turns the outputs of a run (tile summary, building table and stats, ``event_metrics.csv``,
``comparisons.csv``, ``probe_fits.csv``) into the same ids as the expected-values file; ``compare`` lines them up
and marks each PASS, FAIL, MISSING or INFO. Numbers that need confirmatory labels are not in the file before the
plan freeze, so nothing here reads locked labels.
"""

from __future__ import annotations

import platform
from pathlib import Path

import pandas as pd
import yaml

UNLOCKED_ROLES = ("development", "secondary")


class ReproduceError(ValueError):
    pass


def load_expected(path: str | Path) -> pd.DataFrame:
    """Flatten the expected-values file into rows: id, expected, kind, tol, paper."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rows = []
    for group in ("data", "tokens", "auroc", "probes"):
        for key, spec in (raw.get(group) or {}).items():
            rows.append({"id": f"{group}.{key}" if group in ("tokens", "auroc") else key,
                         "expected": spec.get("expected"), "kind": spec.get("kind", "exact"),
                         "tol": spec.get("tol"), "paper": spec.get("paper", group)})
    tol = raw.get("comparison_tol") or {}
    for label, spec in (raw.get("comparisons") or {}).items():
        rows.append({"id": f"cmp.{label}.estimate", "expected": spec["estimate"], "kind": "abs",
                     "tol": tol.get("estimate", 0.005), "paper": "Table IV"})
        for b in ("lower", "upper"):
            rows.append({"id": f"cmp.{label}.{b}", "expected": spec[b], "kind": "abs",
                         "tol": tol.get("bounds", 0.01), "paper": "Table IV"})
        rows.append({"id": f"cmp.{label}.verdict", "expected": spec["verdict"], "kind": "text",
                     "tol": None, "paper": "Table IV"})
    if not rows:
        raise ReproduceError(f"{path}: no expected values")
    return pd.DataFrame(rows)


def _setting_key(encoder: str) -> str:
    return encoder.split("@", 1)[1] if "@" in encoder else encoder


def collect_actual(tile_summary: pd.DataFrame | None = None, buildings: pd.DataFrame | None = None,
                   building_summary: pd.DataFrame | None = None, event_metrics: pd.DataFrame | None = None,
                   comparisons: pd.DataFrame | None = None, probe_fits: pd.DataFrame | None = None,
                   max_iter: int = 5000) -> dict:
    """Values of a run under the ids used in ``paper/expected_values.yaml`` (any input may be missing)."""
    got: dict = {}
    if tile_summary is not None:
        got["tiles.total"] = int(tile_summary["n_tiles"].sum())
        got["tiles.excluded_nodata"] = int(tile_summary["n_excluded_nodata"].sum())
        got["tiles.problems"] = int(tile_summary["n_problem_tiles"].sum())
    if buildings is not None:
        b = buildings
        got["buildings.total"] = int(len(b))
        kept = b[b["excluded_reason"].isna()]
        got["buildings.kept"] = int(len(kept))
        reasons = b["excluded_reason"].value_counts()
        for r in ("building_nodata", "tile_nodata", "empty_footprint"):
            got[f"buildings.excluded.{r}"] = int(reasons.get(r, 0))
        if "reported_area_is_bbox" in b:
            got["area_field_is_bbox.share"] = round(float(b["reported_area_is_bbox"].fillna(False).mean()), 4)
        unlocked = kept[kept["role"].isin(UNLOCKED_ROLES)]
        prev = {}
        for ev, g in unlocked.groupby("event"):
            pos = int(g["damaged"].astype("boolean").fillna(False).sum())
            got[f"positives.{ev}"] = pos
            got[f"kept.{ev}"] = int(len(g))
            if len(g):
                prev[ev] = 100.0 * pos / len(g)
        if prev:
            got["prevalence.min_unlocked_pct"] = round(min(prev.values()), 1)
            got["prevalence.max_unlocked_pct"] = round(max(prev.values()), 1)
    if building_summary is not None:
        s = building_summary.set_index("event")
        for ev, row in s.iterrows():
            for res, col in (("native", "pct_lt_4_tokens_native"), ("target", "pct_lt_4_tokens_target")):
                if col in s.columns and pd.notna(row.get(col)):
                    got[f"tokens.lt4.{res}.{ev}"] = float(row[col])
        if "raster_agreement" in s.columns:
            ra = s.loc[s["role"].isin(UNLOCKED_ROLES), "raster_agreement"].dropna()
            if len(ra):
                got["raster_agreement.min_unlocked"] = float(ra.min())
    if event_metrics is not None:
        m = event_metrics
        for _, r in m.iterrows():
            got[f"auroc.{_setting_key(r['encoder'])}.{r['pooling']}.{r['event']}"] = round(float(r["auroc"]), 3)
        nat = m[(m["encoder"].str.endswith("@native")) & (m["pooling"] == "outline")]
        if len(nat):
            got["auroc.native.outline.mean5"] = round(float(nat["auroc"].mean()), 3)
    if comparisons is not None:
        for _, r in comparisons.iterrows():
            for col in ("estimate", "lower", "upper"):
                got[f"cmp.{r['label']}.{col}"] = round(float(r[col]), 3)
            got[f"cmp.{r['label']}.verdict"] = str(r["verdict"])
    if probe_fits is not None and len(probe_fits):
        got["all_converged"] = bool((probe_fits["n_iter"] < max_iter).all())
    return got


def _judge(row) -> str:
    kind, exp, got = row["kind"], row["expected"], row["got"]
    if kind == "info":
        return "INFO"
    if got is None or (isinstance(got, float) and pd.isna(got)):
        return "MISSING"
    if kind == "exact":
        return "PASS" if got == exp else "FAIL"
    if kind == "text":
        return "PASS" if str(got) == str(exp) else "FAIL"
    if kind == "min":
        return "PASS" if float(got) >= float(exp) else "FAIL"
    if kind == "abs":
        return "PASS" if abs(float(got) - float(exp)) <= float(row["tol"]) + 1e-12 else "FAIL"
    raise ReproduceError(f"unknown kind {kind!r} for {row['id']}")


def compare(expected: pd.DataFrame, got: dict) -> pd.DataFrame:
    """One row per expected value: got, status (PASS / FAIL / MISSING / INFO)."""
    out = expected.copy()
    out["got"] = out["id"].map(lambda k: got.get(k))
    out["status"] = out.apply(_judge, axis=1)
    return out[["id", "paper", "expected", "got", "tol", "kind", "status"]]


def verdict(report: pd.DataFrame) -> str:
    counts = report["status"].value_counts().to_dict()
    checked = sum(counts.get(k, 0) for k in ("PASS", "FAIL", "MISSING"))
    ok = counts.get("PASS", 0) == checked
    return (f"{'REPRODUCED' if ok else 'NOT REPRODUCED'}: {counts.get('PASS', 0)} of {checked} checked values match"
            f" (FAIL {counts.get('FAIL', 0)}, MISSING {counts.get('MISSING', 0)}, INFO {counts.get('INFO', 0)})")


def environment() -> dict:
    """Versions that matter for reproducing the numbers (recorded in every run manifest)."""
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "pyarrow", "rasterio", "torch", "transformers"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        env["gpu"] = None
    return env
