# Pre-registration: DamageSAR-Bench confirmatory analysis

- **Status:** DRAFT. Frozen on **10 Jan 2027** by commit tag `prereg-2027-01-10` (and, ideally, an OSF upload).
- **Design version:** v11.1 (FYP Master, 9 Nov 2026).
- Items marked **TBD** are fixed on development events before the freeze. Anything changed after the freeze is reported as a deviation.

## 1. Scope
Classification of **given building footprints** as damaged-or-destroyed vs intact, from post-event SAR only, on BRIGHT (14 events), holding each disaster out.

## 2. Event roles (`configs/events.yaml`)
- **Confirmatory (6):** Haiti, Türkiye, Morocco, Noto, Bata, Ukraine.
- **Development (5):** Beirut, La Palma, Hawaii, Congo, Libya.
- **Secondary (3):** Marshall, Mexico, Myanmar.
- **Secondary estimate:** every event with valid building units and ≥ N = 50 damaged buildings. Final list and exclusion reasons: TBD.
- **Ukraine units:** BRIGHT instance labels if the March 2026 release covers Ukraine (check: TBD); otherwise open footprints that pass ≥ 80% agreement **on development events**; otherwise Ukraine moves to secondary and the confirmatory set becomes 5 events, with no replacement. Decision: TBD.

## 3. Units and labels
- One building = one instance polygon. Label rule (instance JSON vs raster majority): TBD; mismatch rate reported.
- No-data rule: exclude tiles with ≥ 95% SAR no-data; drop buildings with ≥ 50% no-data footprint. SAR value 0 = no-data.
- Ground resolution and token pooling: ADR 0001, TBD.

## 4. Models
- Core encoders: ImageNet ViT-B, DINOv2 ViT-B/14, DINOv3-LVD ViT-B/16. Optional: one SAR-pretrained encoder (flagged if it may contain BRIGHT tiles).
- Channel adapter (1-channel summed vs 3-channel replicated): TBD on development events.
- Probe: logistic regression, C = 1, class-balanced, no tuning on held-out events.
- **Training rule:** before the freeze, probes train on development + secondary events only; confirmatory labels are not used, even as training data. After the freeze, probes train on all events except the held-out one.

## 5. Primary claims and decision rule
- **Primary confirmatory claims (2):** (1) RQ1 main comparison, DINOv3-LVD vs ImageNet ViT-B; (2) RQ2 projected vs outline pooling, **only if Gate 1 passes**. Everything else is exploratory.
- **Metric:** building-level AUROC, damaged-or-destroyed vs intact. Smallest worthwhile gain: +0.02.
- **Event-level interval:** t-interval over the per-event gains (n − 1 df). Level: 95%, or 97.5% per claim under the multiplicity rule. Multiplicity rule: TBD with supervisor (candidate: Bonferroni split).
- **Decision table** (rows checked in order; first match wins):
  1. Lower end > 0 **and** point estimate ≥ +0.02 → supported: a worthwhile gain.
  2. Lower end > 0, point estimate < +0.02 → small positive effect, below worthwhile.
  3. Lower end ≤ 0 **and** upper end < +0.02 → not a worthwhile gain.
  4. Anything else → inconclusive.
- **Supporting evidence only:** exact sign-flip test (smallest two-sided p = 2/64 with 6 events, 2/32 with 5); Hartung–Knapp random effects across the secondary set.
- Within one event: tile-cluster bootstrap (paired DeLong as a side check).

## 6. Gates
- **Gate 1 (18 Dec 2026, C2):** mean ΔAUROC ≥ +0.02 over the 5 development events, better than outline pooling in ≥ 4 of 5, and better than a random-shift null. Shift rule: TBD (fixed before any AUROC is computed).
- **C3 (early Jan):** relative prevalence error ≤ 50% on ≥ 4 of 5 development events and better than the training-average baseline. Abstain thresholds: TBD.
- **Gate 2:** definition TBD with supervisor (proposed: locked run executed once + new-event mode tested, ~25 Jan 2027).

## 7. Disclosure: what was seen before the freeze
The 31 Oct 2026 audit (tag `audit-2026-10-31`) looked at confirmatory events in four ways: class and instance counts; footprint-to-SAR offsets; raw-signal building AUROC; tile building-count AUROC. No encoder features were computed and no setting was chosen from them.
- **Untouched:** all encoder, probe, C2-gain, C3 and neighbourhood-control results.
- **Partly known:** the image-free building-count control.
- **Shaped a hypothesis:** the 2–27 m offsets motivated C2; C2's settings and gate use development events only.

## 8. Deviations
_None yet._ Any change after the freeze is listed here with its date and reason.
