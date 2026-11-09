# C2 · SAR viewing geometry per event

C2 (layover/shadow-aware evaluation) needs each event's incidence angle, look direction
(left/right) and orbit (ascending/descending). BRIGHT tiles are 8-bit and terrain-corrected and
carry none of this, so it is recovered from the providers' open-data STAC catalogs.

## What the BRIGHT paper gives (arXiv 2501.06019, Table 2 and §2.2)

| event | event date(s) | SAR provider(s) | GSD in paper (m) |
|---|---|---|---|
| beirut | 2020-08-04 | Capella | 1 |
| bata | 2021-03-07 | Capella | 0.5 |
| congo | 2021-05-22 | Capella | 0.33 |
| haiti | 2021-08-14 | Capella | 0.48 |
| la_palma | 2021-09-19 – 2021-12-13 | Capella | 0.30–0.35 |
| marshall | 2021-12-30 – 2022-01-01 | Capella | 0.6 |
| ukraine | 2022-03-22 – 2022-09-21 | Capella | 0.6 |
| turkey | 2023-02-06 | Capella + Umbra | 0.30–0.35 |
| myanmar | 2023-05-14 | Capella | 0.6 |
| hawaii | 2023-08-08 – 2023-08-09 | Capella | 0.6 |
| morocco | 2023-09-08 | Capella | 0.35–0.4 |
| libya | 2023-09-10 | Capella | 0.35 |
| mexico | 2023-10-25 | Capella | 0.35–0.8 |
| noto | 2024-01-01 | Umbra | 0.5 |

Other facts from the paper: single polarisation; Spotlight preferred over Stripmap where the
provider had it; optical and SAR were aligned manually by EO experts (mean RMSE ≈ 1.02 px,
per-event values in its Table C1). The paper gives **no** SAR acquisition dates, modes per event,
incidence angles or look directions. Where the paper's GSD column
differs from the tile geotransforms measured in ADR 0001, the measured values are used.

## How the rest is found

`notebooks/checks/c06_sar_scene_metadata.ipynb` walks the Capella and Umbra STAC catalogs in
Colab (the workspace cannot reach them), and `src/damagesar_bench/c2_metadata.py` applies a
matching rule fixed in advance (area overlap, then date after the event; window −30/+90 days).
Results go to `MyDrive/DamageSAR-Bench/01_data/external/c2_sar_metadata/`; the per-event CSV is
then copied here as `docs/c2/sar_metadata_by_event.csv`.

Status per event: **geometry-ready** (incidence, look direction and orbit all found),
**match-only** (scene found, a field missing), **dilated-only** (no scene; C2 uses dilated
footprints). Only SAR metadata is read, never labels, so confirmatory events are allowed.
