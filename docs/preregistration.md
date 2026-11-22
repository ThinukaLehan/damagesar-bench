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
- One building = one instance polygon from the BRIGHT instance labels (`*_instance_damage.json`). **Label rule:** the damage class is the JSON category (intact / damaged / destroyed, mapped by name). The label-raster majority inside the footprint is computed for unlocked events as an agreement check, and its mismatch rate is reported (p01, 14 Nov: ≈ 0% for every development and secondary event). BRIGHT's COCO `area` field is the bounding-box area, so pixel counts are checked against the polygon's own area instead.
- Ground area uses the true pixel size: UTM as stored; EPSG:4326 converted per axis at the tile's latitude; EPSG:3857 corrected by cos(latitude).
- No-data rule: exclude tiles with ≥ 95% SAR no-data; drop buildings with ≥ 50% no-data footprint. SAR value 0 = no-data.
- Ground resolution: **native** (ADR 0001, decided 20 Nov 2026 on development events only). Token pooling: area-weighted (`pooling/footprint.py`).

## 4. Models
- Core encoders: ImageNet ViT-B, DINOv2 ViT-B/14, DINOv3-LVD ViT-B/16. Optional: one SAR-pretrained encoder (flagged if it may contain BRIGHT tiles).
- Channel adapter (1-channel summed vs 3-channel replicated): TBD on development events.
- Probe: logistic regression, L2, C = 1, class-balanced, no tuning on held-out events. Features are standardised with mean and SD from the training events only; lbfgs, up to 5,000 iterations; a probe that does not converge stops the run (`probe/logistic.py`). Events with `units: none` never train.
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
- **Implementation** (`src/damagesar_bench/stats/`, tested against hand-worked values, printed t tables, a brute-force DeLong, scikit-learn and statsmodels):
  - Verdicts compare values after rounding to 10 decimals, so floating-point noise cannot change a row (e.g. 0.1 − 0.08 counts as exactly +0.02). Equality cases: estimate = +0.02 → row 1; lower = 0 → not "> 0"; upper = +0.02 → not "< +0.02" (inconclusive).
  - AUROC with mid-ranks (ties count ½); average precision as in scikit-learn; DeLong by the Sun & Xu (2014) fast algorithm.
  - Tile-cluster bootstrap: 2,000 replicates, percentile interval, tiles resampled with replacement; replicates with one class are skipped and counted; more than 5% skipped → flagged unreliable.
  - Sign-flip: exact enumeration of all 2ⁿ sign patterns, statistic |sum of gains|.
  - Hartung–Knapp: DerSimonian–Laird τ², t quantile with k − 1 df, no truncation of the HK scale factor.
- **Power (pre-registered method):** minimum detectable mean gain for a two-sided one-sample t-test at 5% with 80% power, solved exactly with the noncentral t distribution: **1.43 SD with 6 events, 1.68 SD with 5** (normal approximation 1.14 SD; quick t approximation 1.42 SD). Pilot-SD rule: SD ≤ 0.020 proceed; 0.020 < SD ≤ 0.035 lead with the estimate; SD > 0.035 lead with the secondary estimate.

## 6. Gates
- **Gate 1 (18 Dec 2026, C2):** mean ΔAUROC ≥ +0.02 over the 5 development events, better than outline pooling in ≥ 4 of 5, and better than a random-shift null. Shift rule: **fixed 22 Nov 2026, before any C2 AUROC was computed** (see *C2 rule* below). **Proposed null rule (to agree):** run the random-shift control R times (same shift-size distribution, different random directions); projected passes if its mean gain is above the 95th percentile of the R random-shift mean gains (R = 1 reduces to "strictly larger"). A gain of exactly 0 does not count as "better". Implemented in `stats/gates.py`.

- **C3 (early Jan):** relative prevalence error ≤ 50% on ≥ 4 of 5 development events and better than the training-average baseline. **Proposed (to agree):** "better" = a lower relative error than the training-average baseline on at least 4 of the same 5 development events. Abstain thresholds: **fixed 22 Nov 2026, before any C3 result** (see *C3 rule* below).
- **Gate 2:** definition TBD with supervisor (proposed: locked run executed once + new-event mode tested, ~25 Jan 2027).

## 7. Disclosure: what was seen before the freeze
The 31 Oct 2026 audit (tag `audit-2026-10-31`) looked at confirmatory events in four ways: class and instance counts; footprint-to-SAR offsets; raw-signal building AUROC; tile building-count AUROC. No encoder features were computed and no setting was chosen from them.
- **Untouched:** all encoder, probe, C2-gain, C3 and neighbourhood-control results.
- **Partly known:** the image-free building-count control.
- **Shaped a hypothesis:** the 2–27 m offsets motivated C2; C2's settings and gate use development events only.

## 8. Deviations
_None yet._ Any change after the freeze is listed here with its date and reason.

## C2 rule (fixed 22 Nov 2026, before any C2 AUROC was computed)

- **C2 as tested at Gate 1 = image-matched projected pooling.** Each tile's footprints are moved by that tile's
  measured footprint-to-SAR offset (`c2_offsets.measure_offsets`: cross-correlation of the rasterised footprints
  with the SAR image at 1/4 resolution within ±40 m, seed 0, 3 permutation-null tiles; no labels are read), **only
  where the offset is real**: peak_z ≥ 3 **and** it beats every null tile. Other tiles are pooled as drawn.
  Pooling is otherwise outline pooling (`pooling: matched`). Encoder: DINOv3 ViT-B/16, native resolution.
- **Why image-matched rather than geometric:** the geometric shift h·cot θ needs building heights, which BRIGHT
  does not provide, and viewing geometry is missing for several events. The measured offset needs neither.
- **Random-shift null:** R = 10 draws (`pooling: rshift:0` … `rshift:9`); each moves every tile by the **same
  distance** as C2 in a uniformly random direction (seeded by tile and draw); tiles C2 does not move are not moved.
- **Gate 1 rule unchanged:** mean gain over outline ≥ +0.02 over the 5 development events, better on ≥ 4 of 5,
  and mean gain above the 95th percentile of the 10 null mean gains (`stats/gates.gate1`).
- **Reported but not part of the gate:** C2 against dilated 3 m and against the 10 m ring.
- **Notebook:** `notebooks/pipeline/g01_gate1_c2_matched_pooling.ipynb`.

## C3 rule (fixed 22 Nov 2026, before any C3 result was computed)

- **Inputs:** the probes' held-out scores (`fold_scores`). For target event e, the source = every other event's
  held-out scores (development + secondary), each from a probe that never saw that event. Target labels are not
  read until the estimate is scored.
- **Estimate:** Platt calibration of the logit score on the source; Saerens EM for the target prior, starting at
  the source prior (reported estimate); binary BBSE (adjusted classify-and-count at threshold 0.5) as a cross-check.
- **Abstain** if the source tpr − fpr at 0.5 is below **0.2**, or EM and BBSE differ by more than **50%** of the
  larger. An abstention gives no count.
- **Baseline:** the training-average prevalence (source share of damaged buildings).
- **Gate (development events):** relative error ≤ 50% on ≥ 4 of 5 events **and** lower than the baseline's on
  ≥ 4 of 5; abstentions count as misses for both. Gated on DINOv3 ViT-B/16 outline (native); ImageNet reported.
- **Code / notebook:** `src/damagesar_bench/recal/prevalence.py`, `notebooks/pipeline/c301_c3_recalibration.ipynb`.
