# Paper plan: DamageSAR-Bench

- **Status:** planning (10 Nov 2026). Writing starts Feb–Mar 2027, alongside the FYP report.
- **Authors:** Thinuka Lehan; supervisor co-authorship to agree. AI is not an author; AI use is disclosed
  in the manuscript as the venue requires (`AI_USE_LOG.md` is the record).
- **Before any submission or preprint, agree with the supervisor:** authorship; whether a preprint or
  submission is allowed before the FYP report is submitted and marked (similarity checks may match the
  report against the paper); the venue's current AI-use policy; time budget (the FYP comes first).

## 1. Venue (decide after Gate 1, 18 Dec 2026)

| Outcome | Venue | Shape |
|---|---|---|
| C2 or C3 works | IEEE JSTARS (Q1, open access, APC ≈ USD 1,800) | Full paper: benchmark + C2 + C3 + toolkit |
| C2 alone works | IEEE GRSL | Letter on projected-footprint pooling |
| Neither works | JSTARS or a benchmark/dataset track | Benchmark + pre-registered negatives + toolkit |
| — | IEEE TGRS | Only with stronger method novelty (C4, out of scope) |

Preprint (arXiv) soon after results if the supervisor and university allow; it also protects priority
(competitor watch below).

## 2. Story in one paragraph (draft)
Building-damage models lose a large share of accuracy on disasters they have not seen, and current
evaluations cannot say why: pixel or tile metrics, optical-drawn outlines that do not line up with SAR,
and no shortcut or leakage control. We build DamageSAR-Bench, a leakage-audited, building-level,
SAR-only benchmark that holds each disaster out, and use it to test (i) how well frozen encoders
classify damage to given footprints on unseen events, (ii) whether moving footprints to where the radar
sees the building (projected pooling) helps, and (iii) whether a new event's damage rate can be
estimated without labels. Claims are pre-registered and made at the event level.

## 3. Outline (IMRaD)
1. **Introduction**: problem, evidence of the generalisation gap, the three-way question
   (representation / evaluation / calibration), contributions C1–C3.
2. **Related work**: BRIGHT and the 2026 BRIGHT Challenge; frozen foundation models on SAR (Shibli et al.);
   neighbourhood shortcut (Guo et al.); layover-aware footprints (Sun, Shahzad & Zhu 2017; Shahzad et al.
   2019); label-shift correction (EM, BBSE); building-level SAR damage (Currie & Reinke).
3. **Data**: BRIGHT events and roles, building units, no-data rule, resolution and CRS handling,
   disclosure of what was seen before the freeze.
4. **Method**: toolkit architecture; pooling (outline, dilated, projected); probes and folds;
   recalibration and abstain rule; controls; statistics (two layers, decision table, power).
5. **Experiments**: development-event results and gates; locked confirmatory results; secondary
   estimate; controls; new-event mode.
6. **Discussion**: what fails where (hazard, resolution, building size), limitations, generalisation.
7. **Conclusion** + data and code availability (repo, splits, exclusion lists; licences).

## 4. Planned figures and tables

| # | Content | Source (when available) |
|---|---|---|
| F1 | Exam-hall overview: why models fail on new events (problem figure) | proposal deck |
| F2 | Toolkit pipeline (7 modules, contracts, guard) | ADR 0002, README |
| F3 | SAR geometry: layover/shadow and projected footprint | C2 notes, c06 metadata |
| F4 | Building size in tokens per event (why pooling matters) | ADR 0001 round 2 (p01) |
| F5 | Per-event AUROC with intervals, encoders × pooling | `bench run` output |
| F6 | Event-level gains with t-interval and verdict (RQ1, RQ2) | `stats/compare.py` |
| F7 | Prevalence estimates vs truth, abstain decisions (RQ3) | C3 |
| T1 | Events, roles, units, resolution, CRS, damage share | building table, ADR 0001 |
| T2 | Controls: image-free, shuffled labels, random shift, neighbourhood | controls |

## 5. Competitor watch (check before writing and before submission)
Guo et al. (arXiv 2610.38162, neighbourhood shortcut); BRIGHT / CrossEarth-SAR group (Chen et al.);
KTH (Shibli et al.); Koshimura group. If a close benchmark appears: lead with C2, C3 and new-event mode.

## 6. Open questions for the paper
- Include the 2025 Challenge events (California wildfire, Jamaica hurricane) as an external check?
  Only if labels are public and licences allow.
- Report 8-bit and dB inputs, or only the one chosen on development events?
- How much of the toolkit description goes in the paper vs a software paper / repo docs.
