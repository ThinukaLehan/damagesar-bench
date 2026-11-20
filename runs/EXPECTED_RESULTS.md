# Reproduction checklist: earlier results to reproduce in the FYP Google account

Every check is rerun from this repo in the FYP Google account. Outputs go to
`MyDrive/DamageSAR-Bench/04_outputs/runs/<date>_<notebook>/` (see `docs/drive_layout.md`), and the
executed notebook is saved to `03_notebooks/checks/` in Drive.
Check notebooks live in `notebooks/checks/` and are named `cNN_<what>.ipynb`.
A rerun **passes** if it matches the expected value within the stated tolerance.

| # | Check | Original run | Notebook | Expected result | Tolerance | GPU | Status (FYP account) |
|---|---|---|---|---|---|---|---|
| 1 | BRIGHT file counts and pairing | 3 Nov (Colab) | c01_data_validation | 4,246 SAR / 4,246 labels; 14 events; tiles per event: haiti 73, turkey 1114, morocco 567, noto 79, bata 107, ukraine 513, beirut 133, la_palma 933, hawaii 65, congo 123, libya 124, marshall 77, mexico 212, myanmar 126 | exact | no | ✅ PASS 14 Nov (c01: 0 problem tiles; all per-event counts match 3 Nov) |
| 2 | Pixel class shares (non-confirmatory events only before the freeze) | 3 Nov | c01_data_validation | Overall 82.69 / 10.68 / 6.64% was computed on all events; before the freeze only development + secondary shares are shown, so compare per event against `bright_event_summary.csv` | 0.01 pp | no | ✅ shown per event in c01 `event_summary.csv` (14 Nov); confirmatory hidden as intended |
| 3 | No-data rule | 3 Nov | c01_data_validation | 20 tiles with ≥ 95% SAR no-data excluded | exact | no | ✅ PASS 14 Nov (20 tiles) |
| 4 | Splits | 3 Nov | c01_data_validation (`bench audit`) | confirmatory 6 / development 5 / secondary 3 | exact | no | ✅ PASS 14 Nov (`splits.json`) |
| 5 | New instance labels coverage | 9 Nov (other account) | c02_instance_label_coverage | 3,029 files; 11 events; Ukraine 0, Mexico 0, Myanmar 0 | exact | no | ✅ PASS 9 Nov |
| 6 | Beirut token size (label components) | 9 Nov (other account) | c03_token_size_and_resolution | 10,801 components; <1 token 7.9%, <4 tokens 29.7%, <9 tokens 54.5%; median 7.85 | exact | no | ✅ PASS 9 Nov |
| 7 | Resolution per event | 9 Nov (other account) | c03_token_size_and_resolution | 0.2–0.6 m; Bata/Congo/Haiti/Noto EPSG:4326; Hawaii EPSG:3857 (table in ADR 0001) | 0.001 m | no | ✅ PASS 9 Nov |
| 8 | DINOv3 ViT-B/16 forward pass | 3 Nov | c04_dinov3_forward_pass | output shape [1, 201, 768] for a 224 px input; through the package interface (14, 14, 768) at 224 px and (64, 64, 768) at 1024 px | exact | T4 | ✅ PASS 14 Nov (shape and interface); timing 10.5 tiles/s at 1024 px fp16 |
| 9 | BRIGHT UNet zero-shot, 5 development events | 3 Nov | c05_unet_zeroshot_reproduction (to write from the 3 Nov notebook) | mIoU Beirut 31.63, Congo 55.97, Hawaii 31.73, La Palma 32.74, Libya 33.25 | 0.01 | T4 | to do |
| 10 | UNet inference speed | 3 Nov | c05_unet_zeroshot_reproduction | ≈ 1.2 tiles/s on T4 | ± 20% | T4 | to do |
| 11 | C2 SAR scene metadata (new, not a rerun) | — | c06_sar_scene_metadata | one row per event with status geometry-ready / match-only / dilated-only; no expected values yet (first run sets them) | — | no | ⚠ first run 10 Nov: 5 events with full geometry; Türkiye label bug; Umbra search to widen; rerun pending |
| 12 | Building table (new, not a rerun) | — | p01_building_table | sets the reference values: 244,976 buildings, 242,842 kept; JSON vs raster agreement ≈ 100% | exact | no | ✅ first run 14 Nov (values in EXPERIMENT_LOG) |
| 13 | Resolution test + benchmark v0 (new, ADR 0001 round 3) | — | p02_resolution_benchmark_v0 | first run sets the values: per-event AUROC for DINOv3 at native and 0.3 m, outline and 3 m dilated; timing per resolution | — | T4 | ready after this PR; needs p01 output |

Not rerun: the 31 Oct full audit (offsets, raw-signal AUROC). It touched confirmatory events in ways the
pre-registration already discloses; repeating it adds no evidence and would look at confirmatory data again.
