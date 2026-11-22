# Pre-registration: DamageSAR-Bench confirmatory analysis

- **Status:** DRAFT for the freeze on **25 Nov 2026** (moved forward from 10 Jan 2027 at the supervisor's request, 22 Nov 2026). Frozen by the git tag `prereg-freeze` and a Zenodo deposit of this file; the date, tag and DOI are in `docs/FREEZE.md`. The locked run (`notebooks/locked/k01_locked_confirmatory_test.ipynb`) refuses to start without them.
- **Design version:** v11.1 (FYP Master, 9 Nov 2026).
- Every item once marked TBD was fixed on development events before the freeze (see the decision record at the end). Anything changed after the freeze is reported as a deviation.

## 1. Scope
Classification of **given building footprints** as damaged-or-destroyed vs intact, from post-event SAR only, on BRIGHT (14 events), holding each disaster out.

## 2. Event roles (`configs/events.yaml`)
- **Confirmatory (5, after the Ukraine decision below):** Haiti, Türkiye, Morocco, Noto, Bata. (Ukraine was listed here until 22 Nov 2026.)
- **Development (5):** Beirut, La Palma, Hawaii, Congo, Libya.
- **Secondary (4):** Marshall, Mexico, Myanmar, Ukraine.
- **Secondary estimate:** dropped (decision 22 Nov 2026). Of the secondary events only Marshall has building units, so a random-effects estimate across them cannot be computed. The exact sign-flip test remains as supporting evidence.
- **Ukraine units:** BRIGHT instance labels if the March 2026 release covers Ukraine (checked 22 Nov 2026: it does not; p01 found no Ukraine instance labels); otherwise open footprints that pass ≥ 80% agreement **on development events**; otherwise Ukraine moves to secondary and the confirmatory set becomes 5 events, with no replacement. **Decision (22 Nov 2026): Ukraine moves to the secondary set.** BRIGHT's instance labels do not cover it and no open-footprint source has passed the agreement test, so the confirmatory set is **5 events: Haiti, Türkiye, Morocco, Noto, Bata**.

## 3. Units and labels
- One building = one instance polygon from the BRIGHT instance labels (`*_instance_damage.json`). **Label rule:** the damage class is the JSON category (intact / damaged / destroyed, mapped by name). The label-raster majority inside the footprint is computed for unlocked events as an agreement check, and its mismatch rate is reported (p01, 14 Nov: ≈ 0% for every development and secondary event). BRIGHT's COCO `area` field is the bounding-box area, so pixel counts are checked against the polygon's own area instead.
- Ground area uses the true pixel size: UTM as stored; EPSG:4326 converted per axis at the tile's latitude; EPSG:3857 corrected by cos(latitude).
- No-data rule: exclude tiles with ≥ 95% SAR no-data; drop buildings with ≥ 50% no-data footprint. SAR value 0 = no-data.
- Ground resolution: **native** (ADR 0001, decided 20 Nov 2026 on development events only). Token pooling: area-weighted (`pooling/footprint.py`).

## 4. Models
- Primary encoders (RQ1): **one DINOv3 model**, named in `docs/FREEZE.md` as `rq1_encoder` (decision 10, below): either DINOv3-LVD ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`, as first planned) or DINOv3-SAT ViT-L/16 (`facebook/dinov3-vitl16-pretrain-sat493m`, fp32, chosen on the development events in p04), against ImageNet ViT-B/16 (`google/vit-base-patch16-224`). If the ViT-L is named, DINOv3 ViT-B/16 vs ImageNet is also reported, as exploratory. Exploratory otherwise: DINOv2 ViT-B/14, DINOv3 ViT-L/16 (web), the fine-tuned ViT-B (b01) and the texture + random forest reference (b02).
- Channel adapter (decided 22 Nov 2026): the single SAR band (8-bit, divided by 255) is replicated to three channels and normalised with each model's published statistics, as in every development run; for an RGB encoder this equals summing the first-layer weights.
- Probe: logistic regression, L2, C = 1, class-balanced, no tuning on held-out events. Features are standardised with mean and SD from the training events only; lbfgs, up to 5,000 iterations; a probe that does not converge stops the run (`probe/logistic.py`). Events with `units: none` never train.
- **Training rule:** before the freeze, probes train on development + secondary events only; confirmatory labels are not used, even as training data. After the freeze, probes train on all events except the held-out one.

## 5. Primary claims and decision rule
- **Primary confirmatory claims (2):** (1) RQ1 main comparison, the DINOv3 model named in `docs/FREEZE.md` vs ImageNet ViT-B; (2) RQ2 projected vs outline pooling, **only if Gate 1 passes** (it did not: g01, 22 Nov 2026, so RQ2 is not run). Everything else is exploratory.
- **Metric:** building-level AUROC, damaged-or-destroyed vs intact. Smallest worthwhile gain: +0.02.
- **Event-level interval:** t-interval over the per-event gains (n − 1 df). **Multiplicity (decided 22 Nov 2026): Bonferroni over the two primary claims: each is tested with a 97.5% interval, and RQ1 keeps 97.5% even if Gate 1 fails and only RQ1 runs.** (`level: 0.975` in `docs/FREEZE.md`.)
- **Decision table** (rows checked in order; first match wins):
  1. Lower end > 0 **and** point estimate ≥ +0.02 → supported: a worthwhile gain.
  2. Lower end > 0, point estimate < +0.02 → small positive effect, below worthwhile.
  3. Lower end ≤ 0 **and** upper end < +0.02 → not a worthwhile gain.
  4. Anything else → inconclusive.
- **Supporting evidence only:** exact sign-flip test (smallest two-sided p = 2/32 with the 5 confirmatory events). The Hartung–Knapp estimate across the secondary set is dropped (§2).
- Within one event: tile-cluster bootstrap (paired DeLong as a side check).
- **Implementation** (`src/damagesar_bench/stats/`, tested against hand-worked values, printed t tables, a brute-force DeLong, scikit-learn and statsmodels):
  - Verdicts compare values after rounding to 10 decimals, so floating-point noise cannot change a row (e.g. 0.1 − 0.08 counts as exactly +0.02). Equality cases: estimate = +0.02 → row 1; lower = 0 → not "> 0"; upper = +0.02 → not "< +0.02" (inconclusive).
  - AUROC with mid-ranks (ties count ½); average precision as in scikit-learn; DeLong by the Sun & Xu (2014) fast algorithm.
  - Tile-cluster bootstrap: 2,000 replicates, percentile interval, tiles resampled with replacement; replicates with one class are skipped and counted; more than 5% skipped → flagged unreliable.
  - Sign-flip: exact enumeration of all 2ⁿ sign patterns, statistic |sum of gains|.
  - Hartung–Knapp: DerSimonian–Laird τ², t quantile with k − 1 df, no truncation of the HK scale factor.
- **Power (pre-registered method):** minimum detectable mean gain for a two-sided one-sample t-test at 5% with 80% power, solved exactly with the noncentral t distribution: **1.43 SD with 6 events, 1.68 SD with 5** (normal approximation 1.14 SD; quick t approximation 1.42 SD). Pilot-SD rule: SD ≤ 0.020 proceed; 0.020 < SD ≤ 0.035 lead with the estimate; SD > 0.035 lead with the secondary estimate.

## 6. Gates
- **Gate 1 (C2; run 22 Nov 2026 on the development events, g01: NOT PASSED, so RQ2 is not run):** mean ΔAUROC ≥ +0.02 over the 5 development events, better than outline pooling in ≥ 4 of 5, and better than a random-shift null. Shift rule: **fixed 22 Nov 2026, before any C2 AUROC was computed** (see *C2 rule* below). **Null rule (fixed 22 Nov 2026):** run the random-shift control R times (same shift-size distribution, different random directions); projected passes if its mean gain is above the 95th percentile of the R random-shift mean gains (R = 1 reduces to "strictly larger"). A gain of exactly 0 does not count as "better". Implemented in `stats/gates.py`.
- **C3 (early Jan):** relative prevalence error ≤ 50% on ≥ 4 of 5 development events and better than the training-average baseline. **Fixed 22 Nov 2026:** "better" = a lower relative error than the training-average baseline on at least 4 of the same 5 development events. Abstain thresholds: **fixed 22 Nov 2026, before any C3 result** (see *C3 rule* below).
- **Gate 2 (decided 22 Nov 2026):** the locked run is executed once, with RQ3 (C3) reported on the confirmatory events. No further threshold.

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

## Decision record (22 Nov 2026)

The supervisor (Dr. Achala Aponsu) asked on 22 Nov 2026 (phone) to proceed now rather than wait for January, and to
receive written recommendations on the open questions. The decisions above follow those recommendations
(`DamageSAR-Bench: Decisions and recommendations for Dr. Aponsu`, 22 Nov 2026) and are **pending his written
confirmation, which is recorded in `docs/FREEZE.md` before the freeze tag is set**.

**Decision 10 (pending his confirmation):** which DINOv3 model enters RQ1. Recommended: the satellite-pretrained
DINOv3 ViT-L/16 (`rq1_encoder: dinov3_vitl16_sat`), which had the highest mean AUROC on the development events (p04);
the alternative is DINOv3 ViT-B/16 as first planned. The choice is written into `docs/FREEZE.md` before the tag is
set, and the paper states that it was made on the development events.

## Development-phase results known at the freeze

All from development and secondary events; no confirmatory label was used. Listed so that a reader can see what
was known when the plan was frozen.

- DINOv3 ViT-B on the development events (p02/p02b, 20 Nov): DINOv3 native outline AUROC 0.646 / 0.849 / 0.504 / 0.604 / 0.498 (mean 0.620).
- Controls and ImageNet (p03, 22 Nov): ImageNet outline mean 0.663; DINOv3 ring mean 0.618; image-free mean 0.430.
- C3 (c301, 22 Nov): gate not passed (DINOv3 abstains on 5 of 5; ImageNet abstains on 3 and fails on 2).
- Gate 1 (g01, 22 Nov): **not passed**. C2 gains −0.010 / +0.006 / −0.010 / +0.040 / +0.011 (mean +0.007; better on 3 of 5); random-shift null 95th percentile +0.016. RQ2 is not run.
- More encoders (p04, 22 Nov): outline AUROC means DINOv2 ViT-B/14 0.570, DINOv3 ViT-L web 0.649, **DINOv3 ViT-L satellite 0.765**; satellite − web +0.117 [−0.033, +0.266], better on 4 of 5. At the probe threshold every encoder's predicted damaged share is far from the true share.
- References: ViT-B fine-tuned on 32 m crops (b01) mean 0.651; texture + random forest (b02) mean 0.598; satellite ViT-L probe − texture +0.167 [+0.040, +0.294], better on 5 of 5 (exploratory).
- Reproduction (r01, 22 Nov): 172 of 172 checks reproduced from raw data.
