import json

import numpy as np
import pandas as pd
import pytest

from damagesar_bench.baselines.finetune import (ENCODER, FinetuneConfig, building_crops, crop_building,
                                                leave_one_event_out)


def test_crop_is_centred_and_padded():
    img = np.zeros((100, 100), np.float32)
    img[45:55, 45:55] = 1.0                                     # bright 10 x 10 px square at the centre
    c = crop_building(img, [[45, 45, 55, 45, 55, 55, 45, 55]], 1.0, 1.0, side_m=20, out_px=20)
    assert c.shape == (20, 20) and c.dtype == np.uint8
    assert c[5:15, 5:15].min() == 255 and c[:5].max() == 0      # building in the middle, dark around it
    edge = crop_building(img, [[0, 0, 4, 0, 4, 4, 0, 4]], 1.0, 1.0, side_m=20, out_px=20)
    assert edge[:8, :8].max() == 0                              # outside the tile is padded with 0


def test_crop_uses_metres_per_axis():
    img = np.ones((200, 200), np.float32)
    c = crop_building(img, [[100, 100, 101, 100, 101, 101]], 0.5, 0.25, side_m=10, out_px=8)
    assert c.shape == (8, 8) and c.min() == 255                  # 20 x 40 px window, all inside


class TinyNet:
    """A minimal torch model: mean brightness -> logit."""

    def __new__(cls, cfg):
        import torch

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(1, 2)

            def forward(self, x):
                return self.lin(x.mean(dim=(1, 2, 3))[:, None])

        return M()


def test_leave_one_event_out_learns_a_trivial_signal():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    rows, crops = [], []
    for ev in ["a", "b", "c"]:
        for i in range(120):
            dmg = bool(rng.random() < 0.4)
            rows.append({"event": ev, "tile_id": f"{ev}_t", "instance_id": i, "role": "development", "damaged": dmg})
            crops.append(np.full((8, 8), 200 if dmg else 40, np.uint8))
    table = pd.DataFrame(rows)
    table["damaged"] = table["damaged"].astype("boolean")
    cfg = FinetuneConfig(epochs=30, lr=0.05, batch=32, out_px=8)
    fs = leave_one_event_out(table, np.stack(crops), ["a", "c"], cfg, model_factory=TinyNet)
    assert set(fs["event"]) == {"a", "c"} and fs["encoder"].eq(ENCODER).all() and fs["pooling"].eq("crop32m").all()
    for ev, g in fs.groupby("event"):
        y = g["damaged"].astype(bool).to_numpy()
        assert g.loc[y, "score"].min() > g.loc[~y, "score"].max()   # perfectly separated
        assert ev not in g["train_events"].iloc[0].split("+")


def test_building_crops_reads_each_tile_once():
    calls = []

    class G:
        px_x_m = px_y_m = 1.0

    def read(t):
        calls.append(t)
        return np.ones((50, 50), np.float32), G()

    b = pd.DataFrame({"tile_id": ["t1", "t1", "t2"], "instance_id": [0, 1, 0], "excluded_reason": [None, None, None],
                      "segmentation_px": [json.dumps([[10, 10, 12, 10, 12, 12]])] * 3})
    table, crops = building_crops(b, read, FinetuneConfig(out_px=16))
    assert calls == ["t1", "t2"] and crops.shape == (3, 16, 16) and len(table) == 3
