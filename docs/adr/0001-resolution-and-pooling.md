# ADR 0001: Ground resolution and token pooling

- **Status:** proposed. Evidence round 1 done (9 Nov 2026); decide after round 2 with instance polygons, by 20 Nov 2026.
- **Date:** 2026-11-09
- **Requirements:** FR5, FR6, NFR3

## Context
ViT-B/16 on a 1024 px tile gives a 64 × 64 token grid. Many buildings cover only a few tokens, and a C2
layover shift of a few metres may not move any token. Resolution also differs between events, so a token
covers different ground areas in different events, which confounds cross-event comparison.

## Evidence round 1 (notebook 02, Colab, 9 Nov 2026)

**Resolution per event** (geotransforms only; no labels read):

| Event | Role | CRS | Resolution (m) | Token side, ViT/16 (m) |
|---|---|---|---|---|
| Bata | confirmatory | EPSG:4326 | 0.51 | 8.1 |
| Haiti | confirmatory | EPSG:4326 | 0.47 | 7.5 |
| Noto | confirmatory | EPSG:4326 | 0.47 | 7.6 |
| Morocco | confirmatory | UTM 29N | 0.31 | 4.9 |
| Türkiye | confirmatory | UTM 37N | 0.31–0.35 | 5.6 |
| Ukraine | confirmatory | UTM 37N | 0.35–0.47 | 7.5 |
| Beirut | development | UTM 36N | 0.50 | 8.0 |
| Congo | development | EPSG:4326 | 0.33 | 5.3 |
| Hawaii | development | EPSG:3857 | 0.38 (Web Mercator; ≈ 0.35 on the ground) | 6.0 |
| La Palma | development | UTM 28N | 0.30 | 4.8 |
| Libya | development | UTM 34N | 0.35 | 5.6 |
| Marshall | secondary | UTM 13N (NAD83) | 0.60 | 9.6 |
| Mexico | secondary | UTM 14N | 0.20–0.30 | 4.8 |
| Myanmar | secondary | UTM 46N | 0.35 | 5.6 |

Findings: resolution ranges 0.2–0.6 m, so one token covers 4.8–9.6 m. Four events are stored in geographic
coordinates (degrees), so all geometry code must convert to metres. Hawaii is in Web Mercator.

**Beirut building size in tokens** (native 0.5 m = 8 m tokens; buildings = connected components of the
label raster, after the no-data rule; 10,801 components):

| | < 1 token | < 4 tokens | < 9 tokens | Median tokens | Median area |
|---|---|---|---|---|---|
| Beirut, native (0.5 m) | 7.9% | 29.7% | 54.5% | 7.85 | 502 m² |

Caveat: connected components merge touching buildings (10,801 components vs ~22,000 instance buildings in
Beirut), so these shares are a **lower bound**; true buildings are smaller.

**C2 shift in tokens at Beirut (8 m tokens):** 2 m = 0.25, 5 m = 0.62, 10 m = 1.25, 15 m = 1.88, 27 m = 3.38.
A shift below ~8 m moves a footprint by less than one token, so at native resolution C2 can only matter for
the larger offsets.

## Options
1. Native resolution, area-weighted pooling over tokens.
2. Resample every tile to one ground resolution, area-weighted pooling.
3. Option 2 plus bilinear upsampling of the token grid (×2) before pooling.
4. Building-centred crops per building (best detail, highest cost).

## Leaning (not yet decided)
At least 30% of Beirut buildings cover fewer than 4 tokens even with merged components, which falls in the
"upsample" band. Leaning: **resample all events to a common 0.3 m** (the finest common resolution; token side
4.8 m), plus area-weighted pooling, with ×2 token upsampling as an ablation. Check the cost with the B5
timing run, because Beirut and Marshall tiles grow by 1.7–2× per side.

## Evidence round 2 (to do)
Run `notebooks/pipeline/p01_building_table.ipynb`: its summary gives the shares under 1/4/9 tokens per event with
the **instance polygons**, at native resolution and at 0.3 m (`pct_lt_k_tokens_native` / `_target`).

Note from building the loader (9 Nov): EPSG:4326 pixels are not square on the ground (Noto, 37.5°N: a pixel square
in degrees is ~0.38 m × 0.48 m), and Hawaii's EPSG:3857 pixels are ~0.352 m on the ground, not the stored 0.377 m.
The resolution table in round 1 used the x side only; the building table uses the true pixel area.

## Decision
_To be filled after round 2._

## Consequences
The choice is fixed in docs/preregistration.md before any Gate 1 AUROC is computed. Pooling code must handle
EPSG:4326 and EPSG:3857 tiles (convert to metres).

## Implementation note (10 Nov 2026)
Area-weighted pooling is implemented in `src/damagesar_bench/pooling/footprint.py` for `outline` and
`dilated:<m>` (true ground metres per axis, so EPSG:4326 tiles are handled). It works for any token grid
that covers the whole tile, so the resolution decision (native vs common 0.3 m) only changes the encoder's
input size, not the pooling code. Buildings smaller than one pixel use the token at their centre and are
flagged (`pool_pixels` = 0); round 2 will report how many there are per event.

