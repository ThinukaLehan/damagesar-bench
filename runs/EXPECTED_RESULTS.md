# Reproduction checklist: earlier results to reproduce in the FYP Google account

Every check is rerun from this repo in the FYP Google account. Outputs go to
`MyDrive/DamageSAR-Bench/04_outputs/runs/<date>_<notebook>/` (see `docs/drive_layout.md`), and the
executed notebook is copied to `03_notebooks/checks/` in Drive.
A rerun **passes** if it matches the expected value within the stated tolerance.

| # | Check | Original run | Notebook | Expected result | Tolerance | GPU |
|---|---|---|---|---|---|---|
| 1 | BRIGHT file counts and pairing | 3 Nov (Colab) | 00_validate_bright | 4,246 SAR / 4,246 labels; 14 events; tiles per event: haiti 73, turkey 1114, morocco 567, noto 79, bata 107, ukraine 513, beirut 133, la_palma 933, hawaii 65, congo 123, libya 124, marshall 77, mexico 212, myanmar 126 | exact | no |
| 2 | Pixel class shares (non-confirmatory events only before the freeze) | 3 Nov | 00_validate_bright | Overall 82.69 / 10.68 / 6.64% was computed on all events; before the freeze only development + secondary shares are shown, so compare per event against `bright_event_summary.csv` | 0.01 pp | no |
| 3 | No-data rule | 3 Nov | 00_validate_bright | 20 tiles with ≥ 95% SAR no-data excluded | exact | no |
| 4 | Splits | 3 Nov | 00_validate_bright (`bench audit`) | confirmatory 6 / development 5 / secondary 3 | exact | no |
| 5 | New instance labels coverage | 9 Nov (other account) | 01_check_instance_labels | 3,029 files; 11 events; Ukraine 0, Mexico 0, Myanmar 0 | exact | no |
| 6 | Beirut token size (label components) | 9 Nov (other account) | 02_token_size_beirut | 10,801 components; <1 token 7.9%, <4 tokens 29.7%, <9 tokens 54.5%; median 7.85 | exact | no |
| 7 | Resolution per event | 9 Nov (other account) | 02_token_size_beirut | 0.2–0.6 m; Bata/Congo/Haiti/Noto EPSG:4326; Hawaii EPSG:3857 (table in ADR 0001) | 0.001 m | no |
| 8 | DINOv3 ViT-B/16 forward pass | 3 Nov | 03_dinov3_forward (to write) | output shape [1, 201, 768] for a 224 px input | exact | T4 |
| 9 | BRIGHT UNet zero-shot, 5 development events | 3 Nov | 04_unet_zeroshot_dev (from the 3 Nov notebook) | mIoU Beirut 31.63, Congo 55.97, Hawaii 31.73, La Palma 32.74, Libya 33.25 | 0.01 | T4 |
| 10 | UNet inference speed | 3 Nov | 04_unet_zeroshot_dev | ≈ 1.2 tiles/s on T4 | ± 20% | T4 |

Not rerun: the 31 Oct full audit (offsets, raw-signal AUROC). It touched confirmatory events in ways the
pre-registration already discloses; repeating it adds no evidence and would look at confirmatory data again.
