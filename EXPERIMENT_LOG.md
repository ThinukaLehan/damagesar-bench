# Experiment log

One line per step. GPU-h and energy are logged for every run (NFR9). Confirmatory results never appear here before the 10 Jan 2027 freeze.

| Date | Step | Run id | Config / commit | GPU-h | Result | Notes |
|---|---|---|---|---|---|---|
| 2026-10-31 | Full BRIGHT audit (headers, counts, offsets) | audit-1 | tag `audit-2026-10-31` | 0 | 15,767 files checked; pixel shares 82.69 / 10.68 / 6.64%; outline-to-SAR offsets 2–27 m | Disclosure: see docs/preregistration.md §7 |
| 2026-11-03 | B3 data check + splits + no-data rule | b3 | Colab | 0 | 4,246 SAR / 4,246 labels / 3,029 JSON; 20 tiles ≥ 95% no-data excluded | |
| 2026-11-03 | B4 UNet zero-shot reproduction (5 dev events) | b4 | Zenodo 15349462 | ≈ 0.3 (T4, ~20 min for 1,378 tiles) | Within 0.01 mIoU of paper Table 10 (Beirut 31.63 vs 31.62) | |
| 2026-11-09 | Repo skeleton: `bench validate`, `bench audit`, guard, tests, CI | — | v0.1.0 | 0 | 27 tests pass, 97% coverage | |
| 2026-11-09 | Checked BRIGHT `cvprw26_trainval_instance_labels.zip` (notebook 01, Colab) | nb01 | Zenodo 19216381 | 0 | 31.2 MB, 3,029 per-tile JSON files (`target_instance_level/<tile>_instance_damage.json`, COCO-style: segmentation, area, bbox, categories) for the same 11 events as before. **Ukraine 0, Mexico 0, Myanmar 0** → Ukraine fallback chain still needed | File names only for confirmatory events; structure read on Beirut |
| 2026-11-09 | Token-size check on Beirut + resolution per event (notebook 02, Colab) | nb02 | — | 0 | Beirut 10,801 label components (lower bound: components merge buildings): <1 token 7.9%, <4 tokens 29.7%, <9 tokens 54.5%, median 7.85 tokens. Resolution 0.2–0.6 m (token side 4.8–9.6 m); Bata, Congo, Haiti, Noto in EPSG:4326, Hawaii in EPSG:3857 | See ADR 0001; round 2 with instance polygons to do |
| 2026-11-09 | **Reproduction in FYP account** (thinukalehanfyp@gmail.com): c02 instance-label coverage (check 5) and c03 token size + resolution (checks 6–7), opened from GitHub in Colab | c02, c03 | main @ 0a189c0 | 0 (CPU) | All identical to the earlier runs: 3,029 files / 11 events / Ukraine-Mexico-Myanmar 0; Beirut 10,801 components, 7.9 / 29.7 / 54.5%, median 7.85; resolution table identical; tile counts per event identical to 3 Nov | Drive: `04_outputs/runs/2026-11-09_01-check-instance-labels/`, `…_02-token-size-beirut/` (named before the cNN rename) |
