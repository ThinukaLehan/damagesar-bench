"""Tests for the building table. Synthetic tiles and COCO-style instance files mimic BRIGHT's format."""

import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

from conftest import write_tif
from damagesar_bench.config import NoDataRule
from damagesar_bench.data import buildings as B

SIZE = 64
CATS = [{"id": 1, "name": "intact", "supercategory": "building"},
        {"id": 2, "name": "damaged", "supercategory": "building"},
        {"id": 3, "name": "destroyed", "supercategory": "building"}]


def rect(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def instance_doc(tile_id, anns, cats=CATS, size=SIZE):
    return {
        "images": [{"id": 0, "width": size, "height": size, "sample_id": tile_id,
                    "geo_reference": {"epsg": 32636}}],
        "annotations": [
            {"id": i, "image_id": 0, "category_id": c, "segmentation": [poly], "iscrowd": 0,
             "area": int(B.polygon_area(poly[0::2], poly[1::2])),
             "bbox": [min(poly[0::2]), min(poly[1::2]), max(poly[0::2]) - min(poly[0::2]),
                      max(poly[1::2]) - min(poly[1::2])]}
            for i, (c, poly) in enumerate(anns)
        ],
        "categories": cats,
    }


def write_instance(folder: Path, tile_id, anns, **kw):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{tile_id}{B.INSTANCE_SUFFIX}").write_text(json.dumps(instance_doc(tile_id, anns, **kw)))


@pytest.fixture
def labels_dir(tmp_path):
    d = tmp_path / "target_instance_level"
    # Beirut (development): intact 10x10, damaged 4x4, destroyed 2x2 (pixels)
    write_instance(d, "beirut-explosion_00000000",
                   [(1, rect(2, 2, 12, 12)), (2, rect(20, 20, 24, 24)), (3, rect(40, 40, 42, 42))])
    # Haiti (confirmatory): two buildings
    write_instance(d, "haiti-earthquake_00000000", [(2, rect(5, 5, 15, 15)), (1, rect(30, 30, 34, 34))])
    return d


# ---------------------------------------------------------------- format handling

def test_category_lookup_maps_names_not_ids():
    swapped = [{"id": 7, "name": "Destroyed"}, {"id": 0, "name": "no damage"}, {"id": 3, "name": "minor_damage"}]
    assert B.category_lookup(swapped) == {7: B.DESTROYED, 0: B.INTACT, 3: B.DAMAGED}


def test_unknown_category_name_stops_the_run():
    with pytest.raises(B.BuildingTableError, match="unknown category"):
        B.category_lookup([{"id": 1, "name": "flooded"}])


def test_tile_id_from_instance_name():
    assert B.tile_id_from_instance_name("target_instance_level/bata-explosion_00000004_instance_damage.json") == \
        "bata-explosion_00000004"
    with pytest.raises(B.BuildingTableError):
        B.tile_id_from_instance_name("bata-explosion_00000004_building_damage.tif")


def test_polygon_area_shoelace():
    assert B.polygon_area([0, 4, 4, 0], [0, 0, 3, 3]) == 12


def test_rasterize_counts_pixel_centres():
    # an axis-aligned rectangle on pixel edges covers exactly w*h pixels
    m = B.rasterize_polygons([rect(2, 3, 7, 9)], 16, 16)
    assert m.sum() == 5 * 6
    assert m[3, 2] and not m[2, 2] and not m[3, 7]


@pytest.mark.parametrize("poly", [rect(2, 3, 7, 9), [1.3, 2.7, 9.8, 1.1, 12.4, 10.6, 3.2, 13.9], rect(0, 0, 16, 16)])
def test_window_rasterize_equals_full_tile(poly):
    full = B.rasterize_polygons([poly], 16, 16)
    win, r0, c0 = B.rasterize_window([poly], 16, 16)
    rebuilt = np.zeros_like(full)
    rebuilt[r0:r0 + win.shape[0], c0:c0 + win.shape[1]] = win
    assert np.array_equal(full, rebuilt)


class _Bounds:
    def __init__(self, top, bottom):
        self.top, self.bottom = top, bottom


def test_pixel_size_projected_is_the_geotransform():
    from rasterio.crs import CRS
    assert B.pixel_size_m(from_origin(0, 0, 0.5, 0.5), CRS.from_epsg(32636), _Bounds(1, 0)) == (0.5, 0.5)
    assert B.metres_per_pixel(from_origin(0, 0, 0.3, 0.3), CRS.from_epsg(32628), _Bounds(1, 0)) == pytest.approx(0.3)


@pytest.mark.parametrize("lat", [0.0, 18.5, 37.5])
def test_pixel_size_geographic_matches_geodesic(lat):
    """EPSG:4326 (Bata, Congo, Haiti, Noto): x and y differ on the ground; checked against pyproj's geodesic."""
    from pyproj import Geod
    from rasterio.crs import CRS
    d = 4.3e-6
    rx, ry = B.pixel_size_m(from_origin(0, lat, d, d), CRS.from_epsg(4326), _Bounds(lat + 1e-3, lat - 1e-3))
    g = Geod(ellps="WGS84")
    gx = g.inv(0, lat, d, lat)[2]
    gy = g.inv(0, lat, 0, lat + d)[2]
    assert rx == pytest.approx(gx, rel=1e-5) and ry == pytest.approx(gy, rel=1e-5)


def test_pixel_size_web_mercator_hawaii():
    """EPSG:3857 (Hawaii): Mercator metres are stretched by 1/cos(lat); true size within 1% of geodesic."""
    from pyproj import Geod, Transformer
    from rasterio.crs import CRS
    x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(-156.68, 20.88)
    rx, ry = B.pixel_size_m(from_origin(x, y, 0.377, 0.377), CRS.from_epsg(3857), _Bounds(y + 200, y - 200))
    inv = Transformer.from_crs(3857, 4326, always_xy=True)
    lo0, la0 = inv.transform(x, y)
    lo1, la1 = inv.transform(x + 0.377, y)
    lo2, la2 = inv.transform(x, y + 0.377)
    g = Geod(ellps="WGS84")
    assert rx == pytest.approx(g.inv(lo0, la0, lo1, la1)[2], rel=0.01)
    assert ry == pytest.approx(g.inv(lo0, la0, lo2, la2)[2], rel=0.01)
    assert rx < 0.36                                        # not the stored 0.377


# ---------------------------------------------------------------- building the table

def test_table_without_rasters(labels_dir, events_cfg, before_freeze):
    df, stats = B.build_building_table(labels_dir, events_cfg, today=before_freeze)
    assert stats.files == 2 and stats.buildings == 5 and len(df) == 5
    beirut = df[df.event == "beirut"].sort_values("instance_id")
    assert list(beirut.damage_class) == [B.INTACT, B.DAMAGED, B.DESTROYED]
    assert list(beirut.damaged) == [False, True, True]
    assert list(beirut.area_px) == [100, 16, 4]
    assert list(beirut.tokens_native) == pytest.approx([100 / 256, 16 / 256, 4 / 256])
    assert not beirut.area_flag.any()
    assert beirut.res_m.isna().all()          # no rasters -> no metric area
    assert beirut.excluded_reason.isna().all()


def test_confirmatory_labels_never_stored_before_freeze(labels_dir, events_cfg, before_freeze):
    df, _ = B.build_building_table(labels_dir, events_cfg, today=before_freeze)
    haiti = df[df.event == "haiti"]
    assert len(haiti) == 2                     # geometry is kept
    assert haiti.labels_locked.all()
    assert haiti.damage_class.isna().all() and haiti.damaged.isna().all()
    s = B.summarise(df).set_index("event")
    assert s.loc["haiti", "class_counts"] == "locked until freeze"
    assert pd.isna(s.loc["haiti", "intact"])


def test_confirmatory_labels_available_after_freeze(labels_dir, events_cfg, after_freeze):
    df, _ = B.build_building_table(labels_dir, events_cfg, today=after_freeze)
    haiti = df[df.event == "haiti"].sort_values("instance_id")
    assert list(haiti.damage_class) == [B.DAMAGED, B.INTACT]


def test_reads_zip_like_bright(labels_dir, tmp_path, events_cfg, before_freeze):
    z = tmp_path / "labels.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p in labels_dir.iterdir():
            zf.write(p, f"target_instance_level/{p.name}")
    df_zip, _ = B.build_building_table(z, events_cfg, today=before_freeze)
    df_dir, _ = B.build_building_table(labels_dir, events_cfg, today=before_freeze)
    assert df_zip.drop(columns=[]).equals(df_dir)


def test_only_events_filter(labels_dir, events_cfg, before_freeze):
    df, stats = B.build_building_table(labels_dir, events_cfg, today=before_freeze, only_events=["beirut"])
    assert set(df.event) == {"beirut"} and stats.files == 1


def test_bad_inputs_raise(tmp_path, events_cfg, before_freeze):
    d = tmp_path / "bad"
    write_instance(d, "beirut-explosion_00000001", [(1, rect(60, 60, 80, 80))])   # outside a 64x64 tile
    with pytest.raises(B.BuildingTableError, match="outside"):
        B.build_building_table(d, events_cfg, today=before_freeze)
    d2 = tmp_path / "bad2"
    write_instance(d2, "beirut-explosion_00000001", [(9, rect(1, 1, 5, 5))])       # undeclared category
    with pytest.raises(B.BuildingTableError, match="undeclared category"):
        B.build_building_table(d2, events_cfg, today=before_freeze)
    d3 = tmp_path / "bad3"
    write_instance(d3, "atlantis-flood_00000000", [(1, rect(1, 1, 5, 5))])          # unknown event
    with pytest.raises(B.BuildingTableError, match="not in events.yaml"):
        B.build_building_table(d3, events_cfg, today=before_freeze)


def test_area_flag_when_reported_area_disagrees(tmp_path, events_cfg, before_freeze):
    d = tmp_path / "flag"
    doc = instance_doc("beirut-explosion_00000002", [(1, rect(0, 0, 10, 10))])
    doc["annotations"][0]["area"] = 300                     # we count 100 pixels
    d.mkdir()
    (d / f"beirut-explosion_00000002{B.INSTANCE_SUFFIX}").write_text(json.dumps(doc))
    df, stats = B.build_building_table(d, events_cfg, today=before_freeze)
    assert df.area_flag.iloc[0] and stats.area_flags == 1


# ---------------------------------------------------------------- with SAR and label rasters

def make_rasters(root: Path, tile_id: str, sar: np.ndarray, label: np.ndarray, res=0.5):
    t = from_origin(500000, 4000000, res, res)
    write_tif(root / "post-event" / f"{tile_id}_post_disaster.tif", sar, transform=t)
    write_tif(root / "target" / f"{tile_id}_building_damage.tif", label, transform=t)


def test_nodata_rules_metres_and_raster_agreement(tmp_path, events_cfg, before_freeze):
    tile = "beirut-explosion_00000000"
    d = tmp_path / "labels"
    write_instance(d, tile, [(1, rect(0, 0, 10, 10)),      # 100 px, half of it no-data -> excluded (>= 50%)
                             (2, rect(20, 20, 30, 30)),    # 100 px, 40 no-data -> kept
                             (3, rect(40, 40, 44, 44))])   # 16 px, clean
    sar = np.full((SIZE, SIZE), 100, dtype=np.uint8)
    sar[0:5, 0:10] = 0                                     # 50 px of building 0
    sar[20:24, 20:30] = 0                                  # 40 px of building 1
    lab = np.zeros((SIZE, SIZE), dtype=np.uint8)
    lab[0:10, 0:10] = 1
    lab[20:30, 20:30] = 3                                  # raster says destroyed, JSON says damaged
    lab[40:44, 40:44] = 3
    root = tmp_path / "bright"
    make_rasters(root, tile, sar, lab, res=0.5)

    df, stats = B.build_building_table(d, events_cfg, bright_root=root, today=before_freeze, target_res_m=0.25)
    df = df.sort_values("instance_id").reset_index(drop=True)
    assert list(df.nodata_frac) == pytest.approx([0.5, 0.4, 0.0])
    assert df.excluded_reason.iloc[0] == "building_nodata"
    assert df.excluded_reason.iloc[1:].isna().all()
    assert stats.buildings_excluded_nodata == 1
    assert list(df.res_m) == pytest.approx([0.5] * 3)
    assert list(df.area_m2) == pytest.approx([25.0, 25.0, 4.0])
    # tokens at 0.25 m with 16-px patches: token = 4 m side = 16 m2
    assert list(df.tokens_at_target) == pytest.approx([25 / 16, 25 / 16, 4 / 16])
    assert list(df.raster_majority_class) == [B.INTACT, B.DESTROYED, B.DESTROYED]
    s = B.summarise(df).set_index("event")
    assert s.loc["beirut", "kept"] == 2
    assert s.loc["beirut", "raster_agreement"] == 0.5     # kept: building 1 disagrees, building 2 agrees


def test_blank_tile_excludes_every_building(tmp_path, events_cfg, before_freeze):
    tile = "beirut-explosion_00000003"
    d = tmp_path / "labels"
    write_instance(d, tile, [(1, rect(0, 0, 10, 10)), (2, rect(20, 20, 30, 30))])
    root = tmp_path / "bright"
    make_rasters(root, tile, np.zeros((SIZE, SIZE), np.uint8), np.zeros((SIZE, SIZE), np.uint8))
    df, stats = B.build_building_table(d, events_cfg, bright_root=root, today=before_freeze)
    assert (df.excluded_reason == "tile_nodata").all() and stats.tiles_excluded_nodata == 1


def test_confirmatory_label_raster_not_read_before_freeze(tmp_path, events_cfg, before_freeze):
    tile = "haiti-earthquake_00000000"
    d = tmp_path / "labels"
    write_instance(d, tile, [(2, rect(0, 0, 10, 10))])
    root = tmp_path / "bright"
    lab = np.full((SIZE, SIZE), 2, dtype=np.uint8)
    make_rasters(root, tile, np.full((SIZE, SIZE), 9, np.uint8), lab)
    df, _ = B.build_building_table(d, events_cfg, bright_root=root, today=before_freeze)
    assert df.raster_majority_class.isna().all() and df.damage_class.isna().all()
    assert df.res_m.iloc[0] == pytest.approx(0.5)          # geometry and metres are still filled


def test_degenerate_polygon_is_empty_footprint_not_nodata(tmp_path, events_cfg, before_freeze):
    tile = "beirut-explosion_00000005"
    d = tmp_path / "labels"
    write_instance(d, tile, [(1, [10.2, 10.2, 10.4, 10.2, 10.4, 10.4]), (1, rect(20, 20, 30, 30))])
    root = tmp_path / "bright"
    make_rasters(root, tile, np.full((SIZE, SIZE), 50, np.uint8), np.zeros((SIZE, SIZE), np.uint8))
    df, stats = B.build_building_table(d, events_cfg, bright_root=root, today=before_freeze)
    df = df.sort_values("instance_id")
    assert df.area_px.iloc[0] == 0 and df.excluded_reason.iloc[0] == "empty_footprint"
    assert pd.isna(df.nodata_frac.iloc[0])
    assert stats.buildings_excluded_nodata == 0


def test_size_mismatch_between_json_and_sar_raises(tmp_path, events_cfg, before_freeze):
    tile = "beirut-explosion_00000004"
    d = tmp_path / "labels"
    write_instance(d, tile, [(1, rect(0, 0, 10, 10))])
    root = tmp_path / "bright"
    make_rasters(root, tile, np.full((32, 32), 9, np.uint8), np.zeros((32, 32), np.uint8))
    with pytest.raises(B.BuildingTableError, match="JSON size"):
        B.build_building_table(d, events_cfg, bright_root=root, today=before_freeze)


def test_parquet_roundtrip_and_cli(labels_dir, tmp_path, events_cfg, before_freeze):
    df, _ = B.build_building_table(labels_dir, events_cfg, today=before_freeze)
    p = B.write_table(df, tmp_path / "out" / "buildings.parquet")
    back = pd.read_parquet(p)
    assert len(back) == len(df) and back.damage_class.isna().sum() == 2
    polys = json.loads(back.segmentation_px.iloc[0])
    assert isinstance(polys, list) and len(polys[0]) == 8

    from damagesar_bench.cli import main
    out = tmp_path / "cli"
    assert main(["buildings", "--labels", str(labels_dir), "--out", str(out), "--today", "2026-11-09",
                 "--events-only", "beirut"]) == 0
    assert (out / "buildings.parquet").exists() and (out / "buildings_summary.csv").exists()
