# DamageSAR-Bench · TODO

Single tracking list for the project. Tick a box (`- [x]`) when done and add the date.
**Owner:** **You** = Thinuka · **Claude** = AI assistant (code/docs; logged in `AI_USE_LOG.md`) · **Sup** = Dr. Achala Aponsu.
**Key dates:** plan freeze **10 Jan 2027** · Gate 1 note **18 Dec 2026** · IPD submission **28 Jan 2027** (aim earlier).
**Rules:** all runs in the FYP Google account (`?authuser=1`, check the email) · Drive under `MyDrive/DamageSAR-Bench/` only · no AI co-author lines in commits · no confirmatory labels before 10 Jan.

Last updated: 9 Nov 2026

---

## 1 · Right now (blocked on you)

- [ ] Create a fine-grained GitHub token (read-only Contents, damagesar-bench only) — **You**
- [ ] Add `GH_TOKEN` to Colab Secrets in the FYP account — **You**
- [ ] Add `HF_TOKEN` (Hugging Face → Settings → Access Tokens → Read) to Colab Secrets — **You**
- [ ] Run **c06_sar_scene_metadata** (CPU runtime) once Colab's runtime outage is fixed, then say "done" — **You** → **Claude** checks results
- [ ] Run **c04_dinov3_forward_pass** (T4) — check 8 — **You**
- [ ] Run **c01_data_validation** — checks 1–4 — **You**
- [ ] Run **p01_building_table** — building table + ADR 0001 round 2 + real label-format check — **You**
- [ ] Copy the 3 Nov files from the old account (`MyDrive/damagesar/data/`) into `DamageSAR-Bench/01_data/derived/` (needed by c01's comparison) — **You**
- [ ] Share the 3 Nov UNet notebook so c05 can be written (checks 9–10) — **You**
- [ ] OK to delete the old "Untitled1" scratch notebook in the other account? — **You** decide

## 2 · Supervisor call (briefing: https://claude.ai/artifact/KAf19x9q4LR5S9bHe4jZaT)

- [ ] Arrange the call and share the briefing doc — **You**
- [ ] D1 Multiplicity: Bonferroni, 97.5% per primary claim — **Sup**
- [ ] D2 Probes train on dev + secondary only before 10 Jan (needed by **24 Nov**) — **Sup**
- [ ] D3 Ukraine agreement test on dev events (needed by **23 Nov**; instance labels do not cover Ukraine) — **Sup**
- [ ] D4 DINOv3 on Ukraine: aggregate-only, or move Ukraine to secondary — **Sup**
- [ ] D5 Gate 2 redefined and moved to ~25 Jan (inside the IPD) — **Sup**
- [ ] Presentation slot date — **Sup**
- [ ] Record outcomes in the briefing's Outcome table + `FYP_Master` — **Claude** (after you report)

## 3 · Results to process after runs (Claude)

- [ ] c06 → check every event, copy `sar_metadata_by_event.csv` into `docs/c2/`, log in `EXPERIMENT_LOG.md`
- [ ] p01 → verify real category names/coordinates, raster agreement, ADR 0001 round 2 → decide resolution + pooling
- [ ] c01 → compare with 3 Nov summary; mark checks 1–4 in `runs/EXPECTED_RESULTS.md`
- [ ] c04 → mark check 8
- [ ] Write c05 UNet reproduction notebook (after you share the 3 Nov notebook)
- [ ] Decide building label source (instance JSON vs raster majority) and report mismatch rate

## 4 · Week 1–2 (9–23 Nov)

- [x] Repo + CI + logs (9 Nov)
- [x] Drive layout (9 Nov)
- [x] Reproduce checks 5–7 in FYP account (9 Nov)
- [x] Building table code + 58 tests (9 Nov)
- [x] C2 matcher + c06 notebook (9 Nov, commit 919b7de)
- [ ] `c2_metadata_by_event` results (needs c06 run)
- [ ] ADR 0001 round 2 (needs p01 run)
- [ ] B5 timing (DINOv3 1024 px fp16)
- [ ] Pre-registration skeleton filled in (`docs/preregistration.md`)
- [ ] Ukraine unit decision path (after D3/D4)
- [ ] Tag the 31 Oct audit commit — **Claude**
- [ ] Fix repo description typos ("building-lvel", "disastes") — **You** on GitHub (or Claude via browser)

## 5 · 24 Nov – 18 Dec (towards Gate 1)

- [ ] Encoder plugins (DINOv3, SAR encoder, baselines)
- [ ] `bench extract` (features to `02_models/features`)
- [ ] Outline / dilated pooling
- [ ] Probes + event-level bootstrap
- [ ] `bench report` v0
- [ ] Pilot SD → physical check of the MDE
- [ ] Projected (layover-aware) pooling for C2 — uses c06 results
- [ ] Random-shift null
- [ ] Benchmark v0 running by **7 Dec** (else move Gate 1 to 9 Jan, not the freeze)
- [ ] **Gate 1 note — 18 Dec**

## 6 · 19 Dec – 10 Jan (towards the freeze)

- [x] Statistics engine, test-first: metrics, DeLong, cluster bootstrap, t-interval, v11.1 decision table, sign-flip, Hartung–Knapp, exact MDE, pilot-SD rule, Gate 1 (9 Nov, moved up from Dec)
- [ ] Agree the Gate 1 random-shift null rule (proposed in `docs/preregistration.md` §6) — **Sup**
- [ ] Define the C3 pass rule precisely ("better than training-event rate": per event or on average?) — **You/Sup**
- [ ] Controls
- [ ] C3 + stress tests + abstain thresholds
- [ ] New-event dry run (`bench new-event`)
- [ ] Final pre-registration
- [ ] **Freeze + git tag — 10 Jan**

## 7 · 11 – 28 Jan (locked run + IPD)

- [ ] Locked confirmatory run (once)
- [ ] New-event mode on dev events
- [ ] **Gate 2 ~25 Jan** (if D5 agreed)
- [ ] IPD document written and checked
- [ ] **IPD submission — 28 Jan** (aim earlier)

## Cut order if behind

FR13 browser → FR14 decoder → SAR encoder → neighbourhood swap → Ukraine to secondary → simpler abstain checks.

## Housekeeping

- [ ] Rename the old Drive run folders to the `cNN_` names — **Claude** (via notebook)
- [ ] Keep `AI_USE_LOG.md` up to date with each AI-assisted change — **Claude**/**You**
