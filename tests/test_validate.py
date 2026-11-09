import numpy as np
import pandas as pd
from rasterio.transform import from_origin

from conftest import SIZE, good_sar, make_tile, simple_label, write_tif
from damagesar_bench.data.validate import validate_dataset, write_reports


def _row(summary: pd.DataFrame, event: str) -> pd.Series:
    return summary.set_index("event").loc[event]


def test_good_dataset_passes(bench_cfg, events_cfg, before_freeze):
    tiles, summary = validate_dataset(bench_cfg, events_cfg, today=before_freeze)
    assert len(tiles) == 4
    assert tiles["ok"].all()
    assert _row(summary, "beirut")["n_tiles"] == 2
    # 64 intact, 16 damaged, 4 destroyed pixels per tile
    assert _row(summary, "beirut")["share_intact"] == round(64 / 84, 4)


def test_blank_tile_is_excluded(bench_cfg, events_cfg, before_freeze):
    tiles, summary = validate_dataset(bench_cfg, events_cfg, today=before_freeze)
    hawaii = tiles[tiles["event"] == "hawaii"].iloc[0]
    assert hawaii["excluded_nodata"]
    assert hawaii["nodata_frac"] == 1.0
    assert _row(summary, "hawaii")["n_excluded_nodata"] == 1


def test_confirmatory_counts_hidden_before_freeze(bench_cfg, events_cfg, before_freeze):
    tiles, summary = validate_dataset(bench_cfg, events_cfg, today=before_freeze)
    haiti = tiles[tiles["event"] == "haiti"].iloc[0]
    assert not haiti["counts_reported"]
    assert pd.isna(haiti["px_damaged"])
    assert _row(summary, "haiti")["class_counts"] == "locked until freeze"
    assert haiti["labels_in_range"]  # schema is still checked


def test_confirmatory_counts_shown_after_freeze(bench_cfg, events_cfg, after_freeze):
    _, summary = validate_dataset(bench_cfg, events_cfg, today=after_freeze)
    assert _row(summary, "haiti")["class_counts"] == "reported"


def test_problems_are_reported(bench_cfg, events_cfg, before_freeze):
    root = bench_cfg.data.root
    bad_label = simple_label()
    bad_label[0, 0] = 9
    make_tile(root, "beirut-explosion_00000002", good_sar(), bad_label)
    # missing label file
    write_tif(root / "post-event" / "beirut-explosion_00000003_post_disaster.tif", good_sar())
    # shifted geotransform on the label
    write_tif(root / "post-event" / "beirut-explosion_00000004_post_disaster.tif", good_sar())
    write_tif(root / "target" / "beirut-explosion_00000004_building_damage.tif", simple_label(),
              transform=from_origin(500010, 4000000, 0.5, 0.5))
    # event not in the config
    make_tile(root, "atlantis-flood_00000000", good_sar(), simple_label())

    tiles, _ = validate_dataset(bench_cfg, events_cfg, today=before_freeze)
    probs = tiles.set_index("tile_id")["problems"]
    assert "label values outside" in probs["beirut-explosion_00000002"]
    assert probs["beirut-explosion_00000003"] == "missing label file"
    assert "geotransforms differ" in probs["beirut-explosion_00000004"]
    assert "not in events.yaml" in probs["atlantis-flood_00000000"]


def test_write_reports(tmp_path, bench_cfg, events_cfg, before_freeze):
    tiles, summary = validate_dataset(bench_cfg, events_cfg, today=before_freeze)
    t, s = write_reports(tiles, summary, tmp_path / "out")
    assert t.exists() and s.exists()
    assert len(pd.read_csv(s)) == len(events_cfg.events)
