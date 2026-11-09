# Experiment log

One line per step. GPU-h and energy are logged for every run (NFR9). Confirmatory results never appear here before the 10 Jan 2027 freeze.

| Date | Step | Run id | Config / commit | GPU-h | Result | Notes |
|---|---|---|---|---|---|---|
| 2026-10-31 | Full BRIGHT audit (headers, counts, offsets) | audit-1 | tag `audit-2026-10-31` | 0 | 15,767 files checked; pixel shares 82.69 / 10.68 / 6.64%; outline-to-SAR offsets 2–27 m | Disclosure: see docs/preregistration.md §7 |
| 2026-11-03 | B3 data check + splits + no-data rule | b3 | Colab | 0 | 4,246 SAR / 4,246 labels / 3,029 JSON; 20 tiles ≥ 95% no-data excluded | |
| 2026-11-03 | B4 UNet zero-shot reproduction (5 dev events) | b4 | Zenodo 15349462 | ≈ 0.3 (T4, ~20 min for 1,378 tiles) | Within 0.01 mIoU of paper Table 10 (Beirut 31.63 vs 31.62) | |
| 2026-11-09 | Repo skeleton: `bench validate`, `bench audit`, guard, tests, CI | — | v0.1.0 | 0 | 27 tests pass, 97% coverage | |
