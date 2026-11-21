# C2 feasibility check: offsets vs token size on the development events (20 Nov 2026)

Question: can projected-footprint pooling (C2) change a building's vector enough to matter, on the five development
events that decide Gate 1 (18 Dec)? Sources: the 31 Oct audit (footprint-vs-SAR cross-correlation and offset
permutation test, `claude/BRIGHT_Validation_and_Novelty_31Oct2026.md`), ADR 0001 rounds 1–3, c06 (10 Nov),
p02/p02b (20 Nov). No new data was read and no C2 AUROC exists yet.

## Per development event

| Event | Token side, native (m) | Offset evidence (31 Oct) | Viewing geometry (c06) | Buildings < 4 tokens | What C2 can do |
|---|---|---|---|---|---|
| Beirut | 8.0 | Largest offset of all events (median best shift ≈ 54 px ≈ 27 m ≈ 3.4 tokens); beyond the permutation null in 5/6 tiles; directions diffuse | none (scene before the Capella catalog) | 41.9% | Shift is several tokens, so pooling can change; but the direction varies, so only a per-tile image-matched shift (option B) can follow it |
| Congo | 5.3 | Beyond the null in 3/4; directions tightly clustered (magnitude not recorded per event) | none | 95.5% | Consistent direction suits a shift, but buildings are tiny: the vector is mostly surroundings whatever the shift |
| Hawaii | 6.0 | **Not distinguishable** from the null | geometry-ready | 32.4% | No measurable offset: expected C2 gain ≈ 0 |
| La Palma | 4.8 | **Not distinguishable** from the null | geometry-ready | 69.8% | No measurable offset: expected C2 gain ≈ 0 |
| Libya | 5.6 | Beyond the null in 6/6; directions diffuse | geometry-ready | 28.4% | Best candidate for a geometry-based shift (offset real, geometry known, buildings larger) |

## What p02/p02b adds
Growing every footprint by 3 m (about half a token) changed AUROC by at most 0.02 in any event and on average by
−0.003 / +0.005 (`paper/CLAIMS_LEDGER.md` L23). A shift smaller than about one token barely changes which tokens a
building touches, so **C2 can only matter where the offset is at least one token (≈ 5–8 m)**.

## Consequence for Gate 1 (as pre-registered)
Gate 1 asks for a mean gain ≥ +0.02 over the five development events **and** a win in at least 4 of 5. Hawaii and
La Palma show no measurable offset, so C2's expected gain there is about zero and each is roughly a coin flip.
Gate 1 can therefore fail for a reason that is about the data (no offset to correct), not about the method.

## Options (for the supervisor; to decide **before** any C2 AUROC is computed)
1. **Keep Gate 1 as written.** Honest, but a likely fail would not show whether C2 works where offsets exist.
2. **Gate on the events with a measurable offset** (Beirut, Congo, Libya; 3 events), fixed now from the 31 Oct
   permutation test, which used no encoder results. Needs "win in ≥ 2 of 3" and the same +0.02 mean.
3. **Measure the offset per tile first (image matching, option B)** and pool on the matched shift only where the
   match is confident; buildings in tiles without a confident match keep outline pooling. This also covers events
   with no viewing geometry (Beirut, Congo, and most confirmatory events).

Recommendation: 3, with the gate restricted as in 2. It needs a small CPU notebook that re-measures the
footprint-vs-SAR offset per development tile (the 31 Oct cross-correlation, now in the repo, with per-event
magnitudes recorded). Whatever is chosen is written into `docs/preregistration.md` §6 before Gate 1 is run.
