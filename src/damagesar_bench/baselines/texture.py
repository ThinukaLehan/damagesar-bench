"""Hand-crafted texture reference: intensity and grey-level co-occurrence (GLCM) features + a random forest.

A cheap, classical reference in the spirit of building-level post-event SAR classification with texture features
and a random forest (Macchiarulo et al., 2025). It answers "do frozen foundation-model features beat what a
classical texture pipeline gets from the same pixels?" on the development events. Exploratory reference only.

* **Regions:** the building footprint (as drawn, rasterized on the native grid) and, as a second setting, the
  footprint plus a 32 m square centred on the building (its context), each described separately.
* **Features per region (15):** intensity mean, SD, 10th/50th/90th percentile and maximum of the 8-bit SAR value
  (scaled to [0, 1]); share of bright pixels (> 0.5) and of dark pixels (< 0.05); the mean relative to the tile's
  median; and seven GLCM statistics (contrast, dissimilarity, homogeneity, angular second moment, energy,
  correlation, entropy) of the region quantized to 32 levels, symmetric, averaged over the four directions at a
  distance of one pixel. No building size or crowding feature is used (that is the image-free control).
* **Classifier (fixed, never tuned on a held-out event):** scikit-learn ``RandomForestClassifier`` with 300 trees,
  ``min_samples_leaf=5``, ``max_features="sqrt"``, ``class_weight="balanced_subsample"``, ``random_state=0``.
  Missing GLCM values (regions too small for any pixel pair) are set to -1. One forest per held-out event,
  trained on every other development and secondary event.
* **Output:** fold scores in the toolkit's contract (``encoder = "texture_rf"``, ``pooling = "footprint"`` or
  ``"footprint_ctx32m"``), so ``stats.compare.event_metrics`` and ``compare`` apply unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..pooling.footprint import building_mask

ENCODER = "texture_rf"
LEVELS = 32
OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1))
GLCM_NAMES = ("contrast", "dissimilarity", "homogeneity", "asm", "energy", "correlation", "entropy")
INTENSITY_NAMES = ("mean", "sd", "p10", "p50", "p90", "max", "bright", "dark", "rel_tile")


@dataclass(frozen=True)
class TextureConfig:
    context_m: float = 32.0
    n_estimators: int = 300
    min_samples_leaf: int = 5
    seed: int = 0


def glcm(q: np.ndarray, mask: np.ndarray, levels: int = LEVELS) -> np.ndarray:
    """Symmetric, normalized GLCM (levels x levels) of the pixels of ``q`` inside ``mask``, four directions, d = 1.

    Only pairs with both pixels inside the mask count. Returns an all-zero matrix if there is no such pair.
    """
    q = np.asarray(q, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    h, w = q.shape
    counts = np.zeros(levels * levels, dtype=np.int64)
    for dy, dx in OFFSETS:
        r0, r1 = 0, h - dy
        c0, c1 = max(0, -dx), w - max(0, dx)
        if r1 <= r0 or c1 <= c0:
            continue
        a, ma = q[r0:r1, c0:c1], mask[r0:r1, c0:c1]
        b, mb = q[r0 + dy:r1 + dy, c0 + dx:c1 + dx], mask[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
        ok = ma & mb
        if ok.any():
            counts += np.bincount((a[ok] * levels + b[ok]).ravel(), minlength=levels * levels)
    p = counts.reshape(levels, levels).astype(np.float64)
    p = p + p.T
    total = p.sum()
    return p / total if total > 0 else p


def glcm_stats(p: np.ndarray) -> dict[str, float]:
    """The seven GLCM statistics; NaN for an empty matrix."""
    if p.sum() <= 0:
        return {k: float("nan") for k in GLCM_NAMES}
    n = p.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    d = i - j
    asm = float((p ** 2).sum())
    mu = float((i * p).sum())
    var = float((((i - mu) ** 2) * p).sum())
    corr = float(((i - mu) * (j - mu) * p).sum() / var) if var > 1e-12 else 1.0
    nz = p[p > 0]
    return {"contrast": float((p * d ** 2).sum()), "dissimilarity": float((p * np.abs(d)).sum()),
            "homogeneity": float((p / (1.0 + d ** 2)).sum()), "asm": asm, "energy": float(np.sqrt(asm)),
            "correlation": corr, "entropy": float(-(nz * np.log2(nz)).sum())}


def region_features(image: np.ndarray, mask: np.ndarray, tile_median: float, prefix: str) -> dict[str, float]:
    """15 features of the pixels of ``image`` (values in [0, 1]) inside ``mask``."""
    v = image[mask]
    if v.size == 0:
        out = {f"{prefix}_{k}": float("nan") for k in INTENSITY_NAMES}
    else:
        mean = float(v.mean())
        out = {f"{prefix}_mean": mean, f"{prefix}_sd": float(v.std()),
               f"{prefix}_p10": float(np.percentile(v, 10)), f"{prefix}_p50": float(np.percentile(v, 50)),
               f"{prefix}_p90": float(np.percentile(v, 90)), f"{prefix}_max": float(v.max()),
               f"{prefix}_bright": float((v > 0.5).mean()), f"{prefix}_dark": float((v < 0.05).mean()),
               f"{prefix}_rel_tile": mean / tile_median if tile_median > 0 else float("nan")}
    q = np.clip(np.floor(image * LEVELS), 0, LEVELS - 1).astype(np.int64)
    out.update({f"{prefix}_{k}": v_ for k, v_ in glcm_stats(glcm(q, mask)).items()})
    return out


def _context_window(polys, geom, side_m: float) -> tuple[int, int, int, int]:
    xs = [v for p in polys for v in p[0::2]]
    ys = [v for p in polys for v in p[1::2]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    hx, hy = side_m / 2 / geom.px_x_m, side_m / 2 / geom.px_y_m
    r0, r1 = max(int(np.floor(cy - hy)), 0), min(int(np.ceil(cy + hy)), geom.height)
    c0, c1 = max(int(np.floor(cx - hx)), 0), min(int(np.ceil(cx + hx)), geom.width)
    return r0, r1, c0, c1


def building_features(polys, image: np.ndarray, geom, tile_median: float, cfg: TextureConfig = TextureConfig()
                      ) -> dict[str, float]:
    """Footprint features (``fp_*``) and context-square features (``ctx_*``) of one building."""
    core, r0, c0 = building_mask(polys, geom)
    if core.size and core.any():
        win = image[r0:r0 + core.shape[0], c0:c0 + core.shape[1]]
        out = region_features(win, core, tile_median, "fp")
    else:
        out = region_features(np.zeros((1, 1), np.float32), np.zeros((1, 1), bool), tile_median, "fp")
    a, b, c, d = _context_window(polys, geom, cfg.context_m)
    ctx = image[a:b, c:d]
    out.update(region_features(ctx, np.ones(ctx.shape, bool), tile_median, "ctx"))
    return out


def texture_table(buildings: pd.DataFrame, read_tile: Callable, cfg: TextureConfig = TextureConfig(),
                  log: Callable[[str], None] = lambda s: None) -> pd.DataFrame:
    """One row per kept building: its keys, label and the 30 texture features."""
    kept = buildings[buildings["excluded_reason"].isna()].sort_values(["tile_id", "instance_id"])
    rows = []
    for i, (tile_id, g) in enumerate(kept.groupby("tile_id", sort=True), 1):
        image, geom = read_tile(tile_id)
        valid = image[image > 0]
        med = float(np.median(valid)) if valid.size else 0.0
        for _, bld in g.iterrows():
            f = building_features(json.loads(bld["segmentation_px"]), image, geom, med, cfg)
            rows.append({"event": bld["event"], "tile_id": tile_id, "instance_id": bld["instance_id"],
                         "role": bld["role"], "damaged": bld["damaged"], **f})
        if i % 100 == 0:
            log(f"{i} tiles done")
    out = pd.DataFrame(rows)
    if len(out):
        out["damaged"] = out["damaged"].astype("boolean")
    return out


def feature_columns(pooling: str) -> list[str]:
    names = list(INTENSITY_NAMES) + list(GLCM_NAMES)
    cols = [f"fp_{k}" for k in names]
    if pooling == "footprint_ctx32m":
        cols += [f"ctx_{k}" for k in names]
    elif pooling != "footprint":
        raise ValueError(f"unknown texture setting {pooling!r}")
    return cols


def leave_one_event_out(table: pd.DataFrame, held_out: list[str], pooling: str = "footprint",
                        cfg: TextureConfig = TextureConfig(), train_roles=("development", "secondary"),
                        log: Callable[[str], None] = lambda s: None) -> pd.DataFrame:
    """Fold scores (toolkit contract) for each held-out event; the forest never sees that event."""
    from sklearn.ensemble import RandomForestClassifier

    cols = feature_columns(pooling)
    x_all = table[cols].to_numpy(np.float64)
    x_all = np.where(np.isfinite(x_all), x_all, -1.0)
    labelled = table["damaged"].notna().to_numpy()
    parts = []
    for ev in held_out:
        tr = labelled & (table["event"] != ev).to_numpy() & table["role"].isin(train_roles).to_numpy()
        te = (table["event"] == ev).to_numpy()
        y = table.loc[tr, "damaged"].astype(bool).to_numpy()
        if y.all() or not y.any():
            raise ValueError(f"held out {ev}: training set needs both classes")
        log(f"--- {pooling}, held out {ev}: train {int(tr.sum())}, test {int(te.sum())}")
        rf = RandomForestClassifier(n_estimators=cfg.n_estimators, min_samples_leaf=cfg.min_samples_leaf,
                                    max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1,
                                    random_state=cfg.seed)
        rf.fit(x_all[tr], y)
        part = table.loc[te, ["event", "tile_id", "instance_id", "role", "damaged"]].copy()
        part["score"] = rf.predict_proba(x_all[te])[:, 1]
        part["encoder"] = ENCODER
        part["pooling"] = pooling
        part["train_events"] = "+".join(sorted(table.loc[tr, "event"].unique()))
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["damaged"] = out["damaged"].astype("boolean")
    return out
