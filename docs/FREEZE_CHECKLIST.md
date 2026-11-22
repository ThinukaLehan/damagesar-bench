# Freeze checklist (draft, 22 Nov 2026)

Everything that must be settled in `docs/preregistration.md` **before** the freeze, with a proposed resolution
and the default if the supervisor gives no other answer. Nothing here uses a confirmatory label.

| # | Open item (plan line) | Proposed resolution | Default |
|---|---|---|---|
| 1 | Freeze date (10 Jan 2027) | Move the freeze to the day the plan is deposited (target Wed 25 Nov 2026); the locked test runs after it, once | Keep 10 Jan |
| 2 | Multiplicity rule (§ statistics) | Bonferroni over the two primary claims: each tested with a 97.5% interval (`level: 0.975`) | Bonferroni |
| 3 | Ukraine units (§ events) | BRIGHT's instance labels do not cover Ukraine, and no open-footprint agreement test has been run on development events, so Ukraine moves to the secondary set: **5 confirmatory events** (power 1.68 SD instead of 1.43 SD) | Secondary; 5 events |
| 4 | Secondary estimate (random effects over the secondary set) | Only Marshall has building units among the secondary events, so the estimate cannot be computed; drop it. Sign-flip p stays as supporting evidence | Dropped |
| 5 | Channel adapter (§ encoders) | Replicate the single SAR band to three channels and normalise with each model's published statistics (as in every run so far); for 3-channel encoders this equals summing the first-layer weights | As used |
| 6 | Gate 1 null rule ("to agree") | R = 10 random-direction draws, 95th percentile (C2 rule section) | As written |
| 7 | C3 "better than baseline" ("to agree") | Lower relative error than the training-average baseline on ≥ 4 of the same 5 development events (C3 rule section) | As written |
| 8 | Gate 2 ("TBD") | Gate 2 = the locked run executed once, with RQ3 (C3) reported on the confirmatory events; no further threshold | As proposed |
| 9 | Encoders (§ encoders) | Primary RQ1 stays DINOv3 ViT-B/16 vs ImageNet ViT-B/16. DINOv2 and the DINOv3 ViT-L web / satellite pair (p04) are exploratory | As proposed |
| 10 | Gate 1 / C3 results | Recorded from g01 and c301 before the freeze; RQ2 runs only if Gate 1 passed | Required |

## Freeze-day steps (in this order)

1. Merge the PR that resolves items 1 to 9 in `docs/preregistration.md`.
2. Deposit `docs/preregistration.md` (as a PDF) on Zenodo (or OSF): title "DamageSAR-Bench pre-registration",
   upload type "Publication / Preprint" or "Other", licence CC BY 4.0. Copy the DOI.
3. One PR that: copies `docs/FREEZE_TEMPLATE.md` to `docs/FREEZE.md` with the date, tag `prereg-freeze`, the DOI
   and `level: 0.975`; and sets `freeze_date` in `configs/events.yaml` to the same date. Merge it.
4. Tag the merge commit: `git tag prereg-freeze <commit> && git push origin prereg-freeze`.
5. Run r01 on the frozen commit (everything still reproduces).
6. Only then run `notebooks/locked/k01_locked_confirmatory_test.ipynb`, **once**.
