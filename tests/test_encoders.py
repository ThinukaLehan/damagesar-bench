import json

import numpy as np
import pandas as pd
import pytest

from damagesar_bench.cli import main
from damagesar_bench.contracts import feature_columns
from damagesar_bench.encoders import Encoder, EncoderError, available, get_encoder, register
from damagesar_bench.encoders.extract import (
    encoder_input_size, extract, read_bright_tile, resample)
from damagesar_bench.encoders.hf import tokens_to_grid
from damagesar_bench.pooling.footprint import TileGeometry

# ------------------------------------------------------------------ interface + registry


def test_registry_lists_builtins():
    names = available()
    assert {"patchstats", "dinov3_vitb16", "dinov2_vitb14", "imagenet_vitb16"} <= set(names)
    with pytest.raises(EncoderError, match="unknown encoder"):
        get_encoder("nope")


def test_new_encoder_is_one_class():
    @register("test_const_encoder")
    class Const(Encoder):
        patch_px, dim = 8, 2

        def _encode(self, image):
            return np.ones((image.shape[0] // 8, image.shape[1] // 8, 2))

    enc = get_encoder("test_const_encoder")
    assert enc.name == "test_const_encoder"
    assert enc.encode(np.zeros((16, 24))).shape == (2, 3, 2)
    with pytest.raises(EncoderError, match="twice"):
        register("test_const_encoder")(Const)


def test_encode_checks_inputs_and_outputs():
    enc = get_encoder("patchstats")
    with pytest.raises(EncoderError, match="single-channel"):
        enc.encode(np.zeros((16, 16, 3)))
    with pytest.raises(EncoderError, match="multiple"):
        enc.encode(np.zeros((20, 16)))
    with pytest.raises(EncoderError, match=r"\[0, 1\]"):
        enc.encode(np.full((16, 16), 2.0))

    class Bad(Encoder):
        name, patch_px, dim = "bad", 16, 2

        def _encode(self, image):
            return np.full((1, 1, 3), np.nan)

    with pytest.raises(EncoderError, match="output shape"):
        Bad().encode(np.zeros((16, 16)))

    class Nan(Bad):
        dim = 3

    with pytest.raises(EncoderError, match="NaN"):
        Nan().encode(np.zeros((16, 16)))


def test_patchstats_hand_worked():
    img = np.zeros((32, 32), np.float32)
    img[:16, :16] = 0.5                 # top-left patch constant 0.5
    img[16:, 16:] = np.tile([0.0, 1.0], (16, 8))   # bottom-right patch: stripes
    t = get_encoder("patchstats").encode(img)
    assert t.shape == (2, 2, 4)
    np.testing.assert_allclose(t[0, 0], [0.5, 0.0, 0.5, t[0, 0, 3]])
    np.testing.assert_allclose(t[1, 1, :3], [0.5, 0.5, 1.0])
    assert t[1, 1, 3] > 0.9             # stripes: |dx| = 1 almost everywhere
    # top-right patch is all zero, but its bottom row's forward difference looks into the stripes
    # below: 8 of its 256 pixels see a jump of 1 -> mean gradient 8/256
    np.testing.assert_allclose(t[0, 1], [0, 0, 0, 8 / 256])


def test_tokens_to_grid_strips_prefix():
    h, w, d = 2, 3, 4
    patches = np.arange(h * w * d, dtype=float).reshape(1, h * w, d)
    for n_prefix in (1, 5):             # ViT/DINOv2 (CLS) and DINOv3 (CLS + 4 registers)
        hidden = np.concatenate([np.full((1, n_prefix, d), -1.0), patches], axis=1)
        grid = tokens_to_grid(hidden, h, w)
        np.testing.assert_array_equal(grid, patches[0].reshape(h, w, d))
    # DINOv3 at 224 px: 201 tokens = 1 + 4 + 14*14 (check 8 in runs/EXPECTED_RESULTS.md)
    assert tokens_to_grid(np.zeros((1, 201, 768)), 14, 14).shape == (14, 14, 768)
    with pytest.raises(EncoderError):
        tokens_to_grid(np.zeros((1, 6 + 7, d)), h, w)
    with pytest.raises(EncoderError):
        tokens_to_grid(np.zeros((2, 7, d)), h, w)

# ------------------------------------------------------------------ resampling


def test_resample_identity_constant_and_ramp():
    img = np.random.default_rng(0).random((10, 12)).astype(np.float32)
    np.testing.assert_array_equal(resample(img, 10, 12), img)
    np.testing.assert_allclose(resample(np.full((10, 10), 0.3), 17, 23), 0.3, atol=1e-7)
    # linear ramp of pixel centres stays linear in the interior when upsampling x2
    ramp = np.tile(np.arange(8, dtype=np.float32), (4, 1))
    up = resample(ramp, 4, 16)
    expected = (np.arange(16) + 0.5) / 2 - 0.5
    np.testing.assert_allclose(up[0, 1:-1], expected[1:-1], atol=1e-6)


@pytest.mark.parametrize("geom, patch, res, expected", [
    (TileGeometry(1024, 1024, 0.5, 0.5), 16, None, (1024, 1024)),
    (TileGeometry(1024, 1024, 0.5, 0.5), 14, None, (1022, 1022)),        # nearest multiple of 14
    (TileGeometry(1024, 1024, 0.5, 0.5), 16, 0.3, (1712, 1712)),         # 1706.7 -> 107 patches
    (TileGeometry(1024, 1024, 0.38, 0.48), 16, 0.3, (1632, 1296)),       # non-square: 1638.4 / 1297.1
])
def test_encoder_input_size(geom, patch, res, expected):
    assert encoder_input_size(geom, patch, res) == expected

# ------------------------------------------------------------------ extract (resumable)


def square(x0, y0, x1, y1):
    return json.dumps([[x0, y0, x1, y0, x1, y1, x0, y1]])


def buildings_for(bright_root, events_cfg):
    rows = []
    for tile, ev, locked in [("beirut-explosion_00000000", "beirut", False),
                             ("beirut-explosion_00000001", "beirut", False),
                             ("haiti-earthquake_00000000", "haiti", True)]:
        for i, (box, dmg) in enumerate([(square(2, 2, 10, 10), True), (square(16, 16, 30, 30), False)]):
            rows.append({"event": ev, "tile_id": tile, "instance_id": i, "role": events_cfg.events[ev].role,
                         "damaged": pd.NA if locked else dmg, "labels_locked": locked, "area_m2": 16.0,
                         "segmentation_px": box, "excluded_reason": None})
    df = pd.DataFrame(rows)
    df["damaged"] = df["damaged"].astype("boolean")
    return df


class CountingEncoder(Encoder):
    name, patch_px, dim = "counting", 16, 4

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = 0, fail_on
        self.inner = get_encoder("patchstats")

    def _encode(self, image):
        self.calls += 1
        if self.fail_on is not None and self.calls == self.fail_on:
            raise KeyboardInterrupt("simulated Colab disconnect")
        return self.inner._encode(image)


def test_read_bright_tile(bright_root):
    img, geom = read_bright_tile(bright_root, "beirut-explosion_00000000")
    assert img.dtype == np.float32 and 0 <= img.min() and img.max() <= 1
    assert geom == TileGeometry(32, 32, 0.5, 0.5)


def test_extract_skips_confirmatory_before_freeze(tmp_path, bright_root, events_cfg, before_freeze):
    b = buildings_for(bright_root, events_cfg)
    df, rep = extract(b, lambda t: read_bright_tile(bright_root, t), CountingEncoder(), ["outline", "dilated:3"],
                      tmp_path / "f", events_cfg, before_freeze)
    assert rep.tiles_total == 2 and rep.tiles_skipped_confirmatory == 1
    assert set(df["event"]) == {"beirut"}
    assert sorted(df["pooling"].unique()) == ["dilated3m", "outline"]
    assert len(df) == 2 * 2 * 2
    assert feature_columns(df) == ["f0000", "f0001", "f0002", "f0003"]
    df2, rep2 = extract(b, lambda t: read_bright_tile(bright_root, t), CountingEncoder(), ["outline"],
                        tmp_path / "g", events_cfg, before_freeze, include_confirmatory=True)
    assert rep2.tiles_total == 3 and df2.loc[df2["event"] == "haiti", "damaged"].isna().all()


def test_extract_resumes_after_interruption(tmp_path, bright_root, events_cfg, before_freeze):
    b = buildings_for(bright_root, events_cfg)
    reader = lambda t: read_bright_tile(bright_root, t)  # noqa: E731
    clean, _ = extract(b, reader, CountingEncoder(), ["outline"], tmp_path / "clean", events_cfg, before_freeze)

    out = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt):
        extract(b, reader, CountingEncoder(fail_on=2), ["outline"], out, events_cfg, before_freeze)
    assert len(list((out / "shards").glob("*.parquet"))) == 1          # first tile kept
    assert not list((out / "shards").glob("*.tmp"))                    # no half-written shard
    enc = CountingEncoder()
    resumed, rep = extract(b, reader, enc, ["outline"], out, events_cfg, before_freeze)
    assert enc.calls == 1 and rep.tiles_done_before == 1 and rep.tiles_encoded == 1
    pd.testing.assert_frame_equal(resumed, clean)


def test_extract_refuses_mixed_settings(tmp_path, bright_root, events_cfg, before_freeze):
    b = buildings_for(bright_root, events_cfg)
    reader = lambda t: read_bright_tile(bright_root, t)  # noqa: E731
    extract(b, reader, CountingEncoder(), ["outline"], tmp_path / "f", events_cfg, before_freeze)
    with pytest.raises(EncoderError, match="other settings"):
        extract(b, reader, CountingEncoder(), ["outline"], tmp_path / "f", events_cfg, before_freeze, target_res_m=0.3)
    with pytest.raises(EncoderError, match="at least one"):
        extract(b, reader, CountingEncoder(), [], tmp_path / "h", events_cfg, before_freeze)


def test_extract_at_target_resolution_pools_on_native_geometry(tmp_path, bright_root, events_cfg, before_freeze):
    b = buildings_for(bright_root, events_cfg)
    reader = lambda t: read_bright_tile(bright_root, t)  # noqa: E731
    df, _ = extract(b, reader, CountingEncoder(), ["outline"], tmp_path / "r", events_cfg, before_freeze,
                    target_res_m=0.25)          # 32 px at 0.5 m -> 64 px input -> 4x4 tokens
    assert len(df) == 4 and np.isfinite(df[feature_columns(df)].to_numpy()).all()


def test_cli_extract(tmp_path, bright_root, events_cfg, capsys):
    b = buildings_for(bright_root, events_cfg)
    path = tmp_path / "buildings.parquet"
    b.to_parquet(path, index=False)
    rc = main(["extract", "--buildings", str(path), "--root", str(bright_root), "--encoder", "patchstats",
               "--pooling", "outline", "--pooling", "dilated:3", "--out", str(tmp_path / "feat"),
               "--today", "2026-11-09"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "confirmatory tiles skipped" in out
    feats = pd.read_parquet(tmp_path / "feat" / "features.parquet")
    assert set(feats["pooling"]) == {"outline", "dilated3m"} and set(feats["encoder"]) == {"patchstats"}
