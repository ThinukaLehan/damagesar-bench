# DamageSAR-Bench · TODO

Single tracking list for the project. Tick a box (`- [x]`) when done and add the date.
**Owner:** **You** = Thinuka · **Claude** = AI assistant (code/docs; logged in `AI_USE_LOG.md`) · **Sup** = Dr. Achala Aponsu.
**Key dates:** plan freeze **10 Jan 2027** · Gate 1 note **18 Dec 2026** · IPD submission **28 Jan 2027** (aim earlier).
**Rules:** every change goes through a branch + pull request that you merge (never push to `main`) · all runs in the FYP Google account (`?authuser=1`, check the email) · Drive under `MyDrive/DamageSAR-Bench/` only · no AI co-author lines in commits · no confirmatory labels before 10 Jan.

Last updated: 14 Nov 2026

---

## 1 · Right now (blocked on you)

- [x] Create a fine-grained GitHub token (read-only Contents, damagesar-bench only) — **You** (10 Nov)
- [x] Add `GH_TOKEN` to Colab Secrets in the FYP account — **You** (10 Nov)
- [x] Add `HF_TOKEN` (Hugging Face → Settings → Access Tokens → Read) to Colab Secrets — **You** (14 Nov)
- [x] Run **c06_sar_scene_metadata** — first run 10 Nov (rerun after the c06 fix PR)
- [x] Run **c04_dinov3_forward_pass** (T4) — check 8 PASS (14 Nov)
- [x] Run **c01_data_validation** — checks 1–4 PASS (14 Nov)
- [x] Run **p01_building_table** — 244,976 buildings; ADR 0001 round 2 (14 Nov)
- [x] Copy the 3 Nov files from the old account into `DamageSAR-Bench/01_data/derived/` (14 Nov)
- [ ] Share the 3 Nov UNet notebook so c05 can be written (checks 9–10) — **You**
- [ ] OK to delete the old "Untitled1" scratch notebook in the other account? — **You** decide

- [x] Merge PRs #1–#8 (all in `main`, 13 Nov)
- [ ] Delete the old branches on GitHub (all their work is in `main`) — **You**, or tell Claude
- [ ] Run **p02_resolution_benchmark_v0** (T4) after its PR is merged — ADR 0001 round 3 + benchmark v0 — **You**
- [ ] Rerun **c06** after the c06-fix PR — **You**
- [ ] Add the 31 Oct audit notebook to the repo so it can be tagged `audit-2026-10-31` — **You** share it, **Claude** adds it

## 2 · Supervisor call (briefing: https://claude.ai/artifact/KAf19x9q4LR5S9bHe4jZaT)

- [ ] Arrange the call and share the briefing doc — **You**
- [ ] D1 Multiplicity: Bonferroni, 97.5% per primary claim — **Sup**
- [ ] D2 Probes train on dev + secondary only before 10 Jan (needed by **24 Nov**) — **Sup**
- [ ] D3 Ukraine agreement test on dev events (needed by **23 Nov**; instance labels do not cover Ukraine) — **Sup**
- [ ] D4 DINOv3 on Ukraine: aggregate-only, or move Ukraine to secondary — **Sup**
- [ ] D5 Gate 2 redefined and moved to ~25 Jan (inside the IPD) — **Sup**
- [ ] Presentation slot date — **Sup**
- [ ] Paper: authorship (supervisor co-author?); is a preprint/submission allowed before the FYP report is marked?; venue AI-use rules; time budget — **Sup** (see `paper/PAPER_PLAN.md`)
- [ ] Questions 6–10 added to the briefing (14 Nov): pooling resolution, C2 main route (image matching), Gate 1 null, C3 pass rule, paper — **Sup**
- [ ] Record outcomes in the briefing's Outcome table + `FYP_Master` — **Claude** (after you report)

## 3 · Results to process after runs (Claude)

- [ ] c06 → fix the coverage label, add the Umbra per-event folder search, rerun, copy the CSV into `docs/c2/` (first run logged 14 Nov)
- [x] p01 → label format OK, raster agreement ≈ 100%, ADR 0001 round 2 written (14 Nov); area check fixed (BRIGHT `area` = bounding box)
- [ ] ADR 0001 round 3 (p02) → decide resolution
- [x] c01 → all per-event tile counts match 3 Nov; checks 1–4 marked (14 Nov)
- [x] c04 → check 8 marked; timing 10.5 tiles/s (14 Nov)
- [ ] Write c05 UNet reproduction notebook (after you share the 3 Nov notebook)
- [x] Building label source: instance JSON; agreement with the label raster ≈ 100% (14 Nov)

## 4 · Week 1–2 (9–23 Nov)

- [x] Repo + CI + logs (9 Nov)
- [x] Drive layout (9 Nov)
- [x] Reproduce checks 5–7 in FYP account (9 Nov)
- [x] Building table code + 58 tests (9 Nov)
- [x] C2 matcher + c06 notebook (9 Nov, commit 919b7de)
- [ ] `c2_metadata_by_event` results (first run 10 Nov; final after the c06 fix)
- [x] ADR 0001 round 2 (14 Nov)
- [ ] B5 timing (DINOv3 1024 px fp16)
- [ ] Pre-registration skeleton filled in (`docs/preregistration.md`)
- [ ] Ukraine unit decision path (after D3/D4)
- [ ] Tag the 31 Oct audit commit — **Claude**
- [ ] Fix repo description typos ("building-lvel", "disastes") — **You** on GitHub (or Claude via browser)

## 5 · 24 Nov – 18 Dec (towards Gate 1)

- [x] Encoder plug-in interface + ImageNet ViT-B, DINOv2, DINOv3 wrappers (10 Nov, PR #4; real-weights check in c04) · SAR encoder still to add
- [x] `bench extract`, resumable per tile (10 Nov, PR #4)
- [x] Outline / dilated pooling, area-weighted (10 Nov, PR #3)
- [x] Probes + event-held-out folds + event metrics + comparison (10 Nov, PR #2)
- [x] `bench run` (features → probes → metrics → verdicts in one command) and `bench report` v0 (10 Nov, PR #6)
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

- [x] Data contracts between stages + PR template (10 Nov, PR #1)
- [ ] Rename the old Drive run folders to the `cNN_` names — **Claude** (via notebook)
- [ ] Add one line to `paper/CLAIMS_LEDGER.md` whenever a result lands — **Claude**
- [ ] Keep `AI_USE_LOG.md` up to date with each AI-assisted change — **Claude**/**You**
