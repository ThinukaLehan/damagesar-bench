"""bench run reads one (encoder, pooling) setting at a time, and features files are written so that it can."""

import pandas as pd
import pyarrow.parquet as pq

from damagesar_bench.encoders.extract import FEATURES_ROW_GROUP, write_features
from damagesar_bench.run import feature_settings, read_setting


def _table(n=50, d=4):
    rows = []
    for enc in ("b", "a"):
        for pool in ("outline", "dilated3m"):
            df = pd.DataFrame({f"f{i:04d}": range(n) for i in range(d)})
            df.insert(0, "encoder", enc)
            df.insert(1, "pooling", pool)
            df.insert(2, "damaged", [i % 3 == 0 for i in range(n)])
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_write_features_sorts_and_uses_small_row_groups(tmp_path, monkeypatch):
    import damagesar_bench.encoders.extract as ex
    monkeypatch.setattr(ex, "FEATURES_ROW_GROUP", 25)
    p = write_features(_table(), tmp_path / "f.parquet")
    back = pd.read_parquet(p)
    assert list(back[["encoder", "pooling"]].drop_duplicates().itertuples(index=False, name=None)) == [
        ("a", "dilated3m"), ("a", "outline"), ("b", "dilated3m"), ("b", "outline")]
    meta = pq.ParquetFile(p).metadata
    assert meta.num_row_groups == 8                      # 200 rows / 25, each group one setting
    assert FEATURES_ROW_GROUP == 8192                    # the module default is unchanged


def test_read_setting_returns_only_that_setting_from_every_file(tmp_path):
    t = _table()
    p1, p2 = tmp_path / "1.parquet", tmp_path / "2.parquet"
    t.iloc[:100].to_parquet(p1, index=False)             # settings b/*
    t.iloc[100:].to_parquet(p2, index=False)             # settings a/*
    assert feature_settings([p1, p2]) == [("a", "dilated3m"), ("a", "outline"), ("b", "dilated3m"), ("b", "outline")]
    block = read_setting([p1, p2], "a", "outline")
    assert len(block) == 50 and set(block["encoder"]) == {"a"} and set(block["pooling"]) == {"outline"}
    assert str(block["damaged"].dtype) == "boolean"
    pd.testing.assert_frame_equal(
        block.drop(columns="damaged").reset_index(drop=True),
        t[(t.encoder == "a") & (t.pooling == "outline")].drop(columns="damaged").reset_index(drop=True))
