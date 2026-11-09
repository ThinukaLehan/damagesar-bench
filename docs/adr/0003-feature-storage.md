# ADR 0003: Store pooled building features, not token grids

- **Status:** proposed
- **Date:** 2026-11-10
- **Requirements:** FR5, FR6, NFR2, NFR3

## Context
A ViT-B/16 token grid for one 1024 px tile is 64 × 64 × 768 values (≈ 6 MB in float16), about 26 GB
for all 4,246 BRIGHT tiles per encoder, and more at 0.3 m. Free Google Drive holds 15 GB. Colab sessions
also disconnect, so extraction must survive interruptions.

## Options
1. Cache token grids, pool later (flexible, too large).
2. Cache token grids for development events only (≈ 9 GB per encoder; still large, and confirmatory
   extraction would need a second design).
3. Encode each tile once, pool it at once with every requested method, keep only building vectors.

## Decision
Option 3. One Parquet shard per tile, written atomically; reruns skip finished tiles; a manifest stops a
resume with different settings. Before the freeze, confirmatory tiles are skipped unless asked for.

## Consequences
Adding a pooling method later (projected pooling for C2) means re-encoding the tiles for that run.
With an encoder at roughly 1 tile/s on a T4, the 5 development events (≈ 1,380 tiles) take under an
hour, inside NFR3. Revisit if C2 needs many pooling variants (then cache development-event tokens on
Colab's local disk for that session only).
