"""Supervised reference: a ViT-B/16 fine-tuned end to end on building crops, leave-one-event-out.

Answers "how far are frozen probes from a model trained for the task?" on the development events. It is a
reference point, not a contribution, and is exploratory.

* **Crop:** a square of ``side_m`` metres on the ground (default 32 m) centred on the building's bounding-box
  centre, cut from the native tile (per-axis pixel size, so EPSG:4326 tiles are handled), padded with 0 where it
  leaves the tile, resized to ``out_px`` (default 112) pixels. The footprint is not drawn on the crop; the crop
  is what a model would see if told "classify the building at the centre".
* **Model:** ``google/vit-base-patch16-224`` (ImageNet-21k + 1k), all weights trained, a 2-class head,
  position embeddings interpolated to the crop size. Input as for the probes: the SAR band replicated to three
  channels, ImageNet mean/SD.
* **Training (fixed, not tuned on any held-out event):** AdamW, lr 2e-5, weight decay 0.05, batch 64,
  ``epochs`` = 2, class-balanced cross-entropy, fp16 autocast on GPU, seed 0. One model per held-out event,
  trained on every other development and secondary event.
* **Output:** fold scores in the toolkit's contract (``encoder = "vitb16_finetuned"``, ``pooling = "crop32m"``),
  so ``stats.compare.event_metrics`` and ``compare`` apply unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

ENCODER = "vitb16_finetuned"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class FinetuneConfig:
    side_m: float = 32.0
    out_px: int = 112
    epochs: int = 2
    lr: float = 2e-5
    weight_decay: float = 0.05
    batch: int = 64
    seed: int = 0
    model_id: str = "google/vit-base-patch16-224"

    @property
    def pooling(self) -> str:
        return f"crop{self.side_m:g}m"


def _bbox_centre(polys) -> tuple[float, float]:
    xs = [v for p in polys for v in p[0::2]]
    ys = [v for p in polys for v in p[1::2]]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def crop_building(image: np.ndarray, polys, px_x_m: float, px_y_m: float, side_m: float, out_px: int) -> np.ndarray:
    """``out_px`` x ``out_px`` uint8 crop of ``side_m`` metres centred on the building (0 outside the tile)."""
    h, w = image.shape
    cx, cy = _bbox_centre(polys)
    half_x, half_y = side_m / 2 / px_x_m, side_m / 2 / px_y_m
    # sample at out_px points per axis (nearest neighbour on the native grid; pixel centres)
    xs = cx - half_x + (np.arange(out_px) + 0.5) * (2 * half_x / out_px)
    ys = cy - half_y + (np.arange(out_px) + 0.5) * (2 * half_y / out_px)
    xi, yi = np.floor(xs).astype(int), np.floor(ys).astype(int)
    vx, vy = (xi >= 0) & (xi < w), (yi >= 0) & (yi < h)
    out = np.zeros((out_px, out_px), dtype=np.float32)
    sub = image[np.ix_(np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1))]
    out[:] = sub * (vy[:, None] & vx[None, :])
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def building_crops(buildings: pd.DataFrame, read_tile: Callable, cfg: FinetuneConfig = FinetuneConfig(),
                   log: Callable[[str], None] = lambda s: None) -> tuple[pd.DataFrame, np.ndarray]:
    """Crops for every kept building (labels unchanged); returns (rows, uint8 array [n, out_px, out_px])."""
    import json

    kept = buildings[buildings["excluded_reason"].isna()].sort_values(["tile_id", "instance_id"])
    rows, crops = [], []
    for i, (tile_id, g) in enumerate(kept.groupby("tile_id", sort=True), 1):
        image, geom = read_tile(tile_id)
        for _, b in g.iterrows():
            crops.append(crop_building(image, json.loads(b["segmentation_px"]), geom.px_x_m, geom.px_y_m,
                                       cfg.side_m, cfg.out_px))
            rows.append(b)
        if i % 100 == 0:
            log(f"{i} tiles cropped")
    table = pd.DataFrame(rows).reset_index(drop=True)
    return table, np.stack(crops) if crops else np.zeros((0, cfg.out_px, cfg.out_px), np.uint8)


def to_tensor_batch(crops: np.ndarray):
    import torch

    x = crops.astype(np.float32)[:, None] / 255.0
    x = np.repeat(x, 3, axis=1)
    x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
    return torch.from_numpy(x)


def default_model(cfg: FinetuneConfig):
    from transformers import AutoModelForImageClassification

    return AutoModelForImageClassification.from_pretrained(cfg.model_id, num_labels=2, ignore_mismatched_sizes=True)


def _forward(model, x):
    try:
        return model(pixel_values=x, interpolate_pos_encoding=True).logits
    except TypeError:          # small test models
        return model(x)


def train_and_score(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, cfg: FinetuneConfig,
                    model_factory: Callable = default_model, device: str | None = None,
                    log: Callable[[str], None] = lambda s: None) -> np.ndarray:
    """Fine-tune on (train_x, train_y); return P(damaged) for test_x."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = model_factory(cfg).to(device)
    y = np.asarray(train_y, dtype=np.int64)
    counts = np.bincount(y, minlength=2).astype(np.float32)
    if (counts == 0).any():
        raise ValueError("training set needs both classes")
    weight = torch.tensor(counts.sum() / (2 * counts), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n = len(y)
    steps = math.ceil(n / cfg.batch)
    model.train()
    for ep in range(cfg.epochs):
        order = rng.permutation(n)
        total = 0.0
        for s in range(steps):
            idx = order[s * cfg.batch:(s + 1) * cfg.batch]
            xb = to_tensor_batch(train_x[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = torch.nn.functional.cross_entropy(_forward(model, xb).float(), yb, weight=weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss)
        log(f"epoch {ep + 1}/{cfg.epochs}: mean loss {total / steps:.4f}")
    model.eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(test_x), 256):
            xb = to_tensor_batch(test_x[s:s + 256]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                p = torch.softmax(_forward(model, xb).float(), dim=1)[:, 1]
            out.append(p.cpu().numpy())
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out) if out else np.zeros(0)


def leave_one_event_out(table: pd.DataFrame, crops: np.ndarray, held_out: list[str], cfg: FinetuneConfig,
                        train_roles=("development", "secondary"), model_factory: Callable = default_model,
                        log: Callable[[str], None] = lambda s: None) -> pd.DataFrame:
    """Fold scores (toolkit contract) for each held-out event; training never sees that event."""
    labelled = table["damaged"].notna().to_numpy()
    parts = []
    for ev in held_out:
        tr = (labelled & (table["event"] != ev).to_numpy() & table["role"].isin(train_roles).to_numpy())
        te = (table["event"] == ev).to_numpy()
        log(f"--- held out {ev}: train {int(tr.sum())}, test {int(te.sum())}")
        score = train_and_score(crops[tr], table.loc[tr, "damaged"].astype(bool).to_numpy(), crops[te], cfg,
                                model_factory, log=log)
        part = table.loc[te, ["event", "tile_id", "instance_id", "role", "damaged"]].copy()
        part["score"] = score
        part["encoder"] = ENCODER
        part["pooling"] = cfg.pooling
        part["train_events"] = "+".join(sorted(table.loc[tr, "event"].unique()))
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["damaged"] = out["damaged"].astype("boolean")
    return out
