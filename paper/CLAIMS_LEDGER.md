# Claims ledger

One row per number or claim the paper (or FYP report) may use. Strength:
**cited** (from another paper) · **descriptive** (counts/measurements, no inference) ·
**exploratory** (development events or non-primary analyses; no significance claims) ·
**confirmatory** (pre-registered primary claims on the locked run, after 10 Jan 2027 only).

| ID | Claim (as it may be written) | Strength | Evidence | Status |
|---|---|---|---|---|
| L01 | On BRIGHT, event-level mIoU of the same 7 models falls from 47.7–52.3% (standard split) to 34.1–39.4% when the event is held out | cited | Chen et al. 2025, ESSD 17, Tables 6 and 10 | verified 5 Nov 2026 |
| L02 | BRIGHT optical–SAR co-registration error ≈ 1.02 px RMSE overall | cited | Chen et al. 2025, §2.3 / Table 4 | verified 9 Nov 2026 |
| L03 | BRIGHT UNet zero-shot results reproduced within 0.01 mIoU on the 5 development events (e.g. Beirut 31.63 vs 31.62) | descriptive | `EXPERIMENT_LOG.md` 2026-11-03 (b4); checkpoints Zenodo 15349462 | done; rerun in FYP account pending (checks 9–10) |
| L04 | BRIGHT pixel class shares 82.69 / 10.68 / 6.64% (intact / damaged / destroyed), matching the paper | descriptive | 31 Oct audit; `EXPERIMENT_LOG.md` | done; rerun via c01 pending |
| L05 | Footprint-to-SAR offsets of 2–27 m, beyond a permutation check in 9 of 14 events; cause (layover vs misalignment) not yet established | descriptive (motivated C2; disclosed) | 31 Oct audit, tag `audit-2026-10-31` (to create) | done; wording must keep "cause not established" until the C2 physical check |
| L06 | BRIGHT ground resolution 0.2–0.6 m; ViT/16 tokens cover 4.8–9.6 m; 4 events stored in EPSG:4326 and 1 in EPSG:3857 | descriptive | ADR 0001 round 1; c03 (reproduced 9 Nov in FYP account) | done |
| L07 | Beirut: ≥ 29.7% of label components cover < 4 ViT/16 tokens at native resolution (lower bound) | descriptive | ADR 0001 round 1; c03 | round 2 with instance polygons pending (p01) |
| L08 | BRIGHT instance labels (Zenodo 19216381) cover 11 of 14 events; none for Ukraine, Mexico, Myanmar | descriptive | c02, `EXPERIMENT_LOG.md` 2026-11-09 | done |
| L09 | With 6 events, two-sided 5%, 80% power, the minimum detectable mean gain is 1.43 SD (1.68 SD with 5 events); the smallest sign-flip p is 2/64 (2/32 with 5) | descriptive (design) | `stats/event_level.py` (`mde`, `sign_flip_test`), tests incl. power simulation | done |
| L10 | The toolkit refuses confirmatory labels before the freeze at every stage hand-over | descriptive (software) | `contracts.py`, `tests/test_contracts.py`; guard tests | done (PR #1) |
| L11 | RQ1: DINOv3-LVD vs ImageNet ViT-B, event-level gain and verdict | confirmatory | locked run | after 10 Jan 2027 |
| L12 | RQ2: projected vs outline pooling (only if Gate 1 passes) | confirmatory | Gate 1 note; locked run | Gate 1 18 Dec 2026 |
| L13 | RQ3: relative prevalence error on development events; abstain rate | exploratory | C3 | late Dec 2026 |
| L14 | RQ4: model–control gaps, isolated-damage recall | exploratory | controls | Dec 2026 – Jan 2027 |
| L15 | Per-event SAR viewing geometry (incidence angle, look direction, orbit) for C2 | descriptive | c06, `docs/c2/` | waiting for the c06 run |
