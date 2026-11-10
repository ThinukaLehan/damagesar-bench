# ADR 0002: Data contracts between pipeline stages

- **Status:** proposed
- **Date:** 2026-11-10
- **Requirements:** FR3, FR5–FR7, FR10, NFR1, NFR5

## Context
Pooling, probes, statistics and the report will be built and reviewed as separate pull requests.
They need a fixed format to exchange data, so each part can be tested alone and a mistake in one
stage stops the run instead of silently changing a result. The confirmatory guard (no confirmatory
labels before 10 Jan 2027) and the probe-training rule (FYP Master §5.3) must also hold at every
hand-over, not only where labels are first read.

## Options
1. Pass plain DataFrames and trust each stage.
2. A heavy schema library (pandera, pydantic) for every table.
3. Three small validators in one module (`contracts.py`), called by every stage on write and read.

## Decision
Option 3: three tables with fixed columns, checked by `validate_building_features`,
`validate_fold_scores` and `validate_event_metrics`.

| Table | Producer → consumer | Key | Main columns |
|---|---|---|---|
| Building features | pooling → probes | encoder, pooling, event, tile_id, instance_id | role, damaged, labels_locked, area_m2, `f0000…` |
| Fold scores | probes → statistics | encoder, pooling, event, tile_id, instance_id | role, damaged, score, train_events (sorted, `+`-joined) |
| Event metrics | statistics → report | encoder, pooling, event | n, n_damaged, auroc, ap, auroc_se, boot_lower/upper, boot_reliable |

Checks enforced:
- keys complete and unique; roles match `configs/events.yaml`; feature columns contiguous from `f0000` and finite;
- before the freeze, confirmatory rows carry no label and `labels_locked` is true; other rows must have labels;
- a held-out event is never in its own training set; before the freeze no confirmatory event is in any training set;
- every event-metrics row has both classes, AUROC and AP in [0, 1], and an ordered bootstrap interval.

## Consequences
Every later PR reads and writes these tables through the validators. A table written before the freeze
fails validation after it (`labels_locked` no longer matches), so features are rebuilt after 10 Jan instead
of reusing stale label columns. Revisit if a stage needs extra columns (add them; required columns stay).
