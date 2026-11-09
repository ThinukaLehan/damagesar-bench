"""Find SAR / label tile pairs on disk (FR1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from damagesar_bench.config import DataConfig
from damagesar_bench.data.events import event_key_from_filename, tile_id_from_filename


@dataclass(frozen=True)
class TilePair:
    tile_id: str
    event: str
    sar_path: Path | None
    label_path: Path | None

    @property
    def paired(self) -> bool:
        return self.sar_path is not None and self.label_path is not None


def discover_pairs(cfg: DataConfig) -> list[TilePair]:
    """Match ``post-event/<id>_post_disaster.tif`` with ``target/<id>_building_damage.tif``.

    Unmatched files are returned too (with one path missing) so validation can report them.
    """
    sar_dir = cfg.root / cfg.sar_dir
    label_dir = cfg.root / cfg.label_dir
    if not sar_dir.is_dir():
        raise FileNotFoundError(f"SAR folder not found: {sar_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label folder not found: {label_dir}")

    sar = {tile_id_from_filename(p.name, cfg.sar_suffix): p for p in sar_dir.glob(f"*{cfg.sar_suffix}")}
    lab = {tile_id_from_filename(p.name, cfg.label_suffix): p for p in label_dir.glob(f"*{cfg.label_suffix}")}

    pairs = []
    for tid in sorted(set(sar) | set(lab)):
        pairs.append(
            TilePair(
                tile_id=tid,
                event=event_key_from_filename(tid),
                sar_path=sar.get(tid),
                label_path=lab.get(tid),
            )
        )
    return pairs
