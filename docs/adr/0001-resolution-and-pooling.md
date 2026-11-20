# ADR 0001: Ground resolution and token pooling

- **Status:** accepted (20 Nov 2026): native resolution. Rounds 1 (9 Nov), 2 (14 Nov) and 3 (20 Nov, `p02` + `p02b`) done.
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

## Evidence round 2 (p01, 14 Nov 2026): instance polygons
Share of kept buildings (BRIGHT instance polygons, after the no-data rule) that cover fewer than 1 / 4 / 9 ViT/16
tokens, at native resolution and resampled to 0.3 m. Geometry only: no confirmatory labels are used.

| Event | Role | Native < 1 / < 4 / < 9 tokens (%) | At 0.3 m < 1 / < 4 / < 9 tokens (%) |
|---|---|---|---|
| Beirut | development | 8.0 / 41.9 / 78.5 | 1.1 / 13.3 / 34.2 |
| Congo | development | 33.1 / 95.5 / 99.3 | 27.3 / 91.8 / 98.9 |
| Hawaii | development | 7.5 / 32.4 / 73.4 | 4.3 / 24.5 / 51.0 |
| La Palma | development | 29.7 / 69.8 / 87.3 | 29.8 / 69.8 / 87.3 |
| Libya | development | 5.3 / 28.4 / 62.4 | 2.9 / 20.4 / 46.1 |
| Marshall | secondary | 22.6 / 91.4 / 98.0 | 2.6 / 22.6 / 70.9 |
| Bata | confirmatory | 20.7 / 84.0 / 97.3 | 4.0 / 30.7 / 73.7 |
| Haiti | confirmatory | 33.8 / 88.8 / 98.7 | 6.5 / 53.9 / 85.1 |
| Morocco | confirmatory | 6.7 / 33.2 / 73.0 | 6.5 / 32.1 / 71.5 |
| Noto | confirmatory | 25.8 / 90.6 / 97.7 | 4.0 / 51.4 / 87.9 |
| Türkiye | confirmatory | 3.8 / 26.0 / 64.9 | 3.3 / 22.4 / 60.3 |

Findings:
1. Round 1 was indeed a lower bound: with instance polygons, 41.9% of Beirut buildings cover fewer than 4 tokens
   (29.7% with merged label components).
2. A common 0.3 m cuts the share under 4 tokens by 1.6–4× for the coarser events (Beirut 41.9 → 13.3%, Marshall
   91.4 → 22.6%, Bata, Haiti, Noto) and leaves the events already near 0.3 m almost unchanged (La Palma, Morocco, Türkiye).
3. Congo and La Palma keep 70–92% of buildings under 4 tokens even at 0.3 m: their buildings are small. For them a
   building's vector is mostly its surroundings at any affordable resolution; this limits C2 and raises the
   neighbourhood-shortcut risk (RQ4).
4. 0.15 m for whole tiles is not affordable on a T4: a 0.5 m tile becomes ~3,400 px (~45,000 tokens), roughly
   100 times the attention cost of a 1024 px tile. If finer detail is needed, crops around buildings (option 4)
   are the realistic route.

## Evidence round 3 (p02 + p02b, 20 Nov 2026): does resolution change AUROC?
DINOv3 ViT-B/16 (frozen) on the development and secondary events, 87,886 buildings per setting; probes per
held-out development event, trained on the other development events + Marshall. Exploratory (development only).
Features: `02_models/features/2026-11-20_dinov3_resolution/`; run: `04_outputs/runs/2026-11-20_p02b-resolution-benchmark-resume/`.

Building-level AUROC per held-out event:

| Setting | Beirut | Congo | Hawaii | La Palma | Libya | Mean |
|---|---|---|---|---|---|---|
| native, outline | 0.646 | 0.849 | 0.504 | 0.604 | 0.498 | 0.620 |
| native, dilated 3 m | 0.651 | 0.853 | 0.501 | 0.605 | 0.478 | 0.618 |
| 0.3 m, outline | 0.531 | 0.762 | 0.576 | 0.598 | 0.486 | 0.591 |
| 0.3 m, dilated 3 m | 0.543 | 0.774 | 0.566 | 0.598 | 0.497 | 0.596 |

Event-level comparison, 0.3 m vs native (5 events, 95% t-interval, 4 df): outline −0.030 [−0.122, +0.062],
dilated −0.022 [−0.110, +0.066]; both **inconclusive**; SD of per-event gains 0.074 / 0.071. Per event, 0.3 m
loses 0.08–0.12 in Beirut and Congo and gains 0.06–0.07 in Hawaii.

Cost (T4, file reading and pooling included, 1,026 tiles): native 506 s (2.03 tiles/s), 0.3 m 738 s (1.39 tiles/s).

Note from building the loader (9 Nov): EPSG:4326 pixels are not square on the ground (Noto, 37.5°N: a pixel square
in degrees is ~0.38 m × 0.48 m), and Hawaii's EPSG:3857 pixels are ~0.352 m on the ground, not the stored 0.377 m.
The resolution table in round 1 used the x side only; the building table uses the true pixel area.

## Decision (20 Nov 2026)
**Native resolution.** Resampling every tile to a common 0.3 m does not improve AUROC on the development events
(inconclusive, point estimates favour native), its effect changes sign between events, and it costs ~1.5× the
extraction time. Round 2's concern (many buildings under 4 tokens) stands and is handled by reporting it per event
and by the neighbourhood control (RQ4), not by resampling. Revisit only with building crops (option 4) if C2 needs
finer detail.

## Consequences
The choice is fixed in docs/preregistration.md before any Gate 1 AUROC is computed. Pooling code must handle
EPSG:4326 and EPSG:3857 tiles (convert to metres).

## Implementation note (10 Nov 2026)
Area-weighted pooling is implemented in `src/damagesar_bench/pooling/footprint.py` for `outline` and
`dilated:<m>` (true ground metres per axis, so EPSG:4326 tiles are handled). It works for any token grid
that covers the whole tile, so the resolution decision (native vs common 0.3 m) only changes the encoder's
input size, not the pooling code. Buildings smaller than one pixel use the token at their centre and are
flagged (`pool_pixels` = 0); round 2 will report how many there are per event.

