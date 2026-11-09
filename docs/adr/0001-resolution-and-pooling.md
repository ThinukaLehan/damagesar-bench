# ADR 0001: Ground resolution and token pooling

- **Status:** proposed (decide by 20 Nov 2026 from notebook 02 on Beirut)
- **Date:** 2026-11-09
- **Requirements:** FR5, FR6, NFR3

## Context
ViT-B/16 on a 1024 px tile gives a 64 × 64 token grid. One token covers 16 px = 5–16 m at BRIGHT's
0.3–1 m resolution, so many buildings cover only a few tokens, and a C2 layover shift of a few metres
may not move any token. Resolution also differs between events, so a token covers different ground
areas in different events, which confounds cross-event comparison.

## Options
1. Native resolution, area-weighted pooling over tokens.
2. Resample every tile to one ground resolution (e.g. 0.5 m), area-weighted pooling.
3. Option 2 plus bilinear upsampling of the token grid (×2) before pooling.
4. Building-centred crops per building (best detail, highest cost).

## Evidence to collect (notebook 02, Beirut, development event)
- Share of buildings covering < 1, < 4, < 9 tokens at native resolution and at 0.5 m.
- Share of buildings whose expected C2 shift (2–27 m) moves the footprint by ≥ 1 token.
- Feature-extraction time per tile on a T4 (NFR3 budget ≤ 25 T4-h).

## Decision
_To be filled._

## Consequences
_To be filled._ The choice is fixed in docs/preregistration.md before any Gate 1 AUROC is computed.
