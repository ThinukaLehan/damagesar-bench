# Brief for a paper-writing session

Paste this note (or point to this file) at the start of a new session that works on the paper.

## Read first, in this order
1. `paper/PAPER_PLAN.md`: venue, outline, figures.
2. `paper/CLAIMS_LEDGER.md`: the only allowed source of numbers and claims.
3. `docs/preregistration.md`: primary claims, decision table, interval method, power.
4. `EXPERIMENT_LOG.md` and `docs/adr/`: what was run and decided, with dates.
5. FYP Master (claude.ai project doc `claude/FYP_Master.md`): background, literature, contributions.

## Rules
- Every number comes from the ledger. If a needed number is not there, stop and ask; do not compute or
  recall it.
- Keep the ledger's strength: descriptive and exploratory results get no significance language;
  confirmatory claims use the pre-registered decision-table wording ("supported", "small positive effect",
  "not a worthwhile gain", "inconclusive"). Never call a non-significant result "no benefit".
- Claims are event-level. Within-event intervals describe one event only.
- No confirmatory numbers before the locked run (after 10 Jan 2027).
- Cite preprints as preprints; verify every reference (authors, year, venue, DOI/arXiv ID) against the source.
- Ukraine: aggregate reporting only; no building-level maps of conflict areas.
- Thinuka Lehan is the author; no AI author or co-author lines in commits or pull requests. AI use is disclosed
  only where the venue requires and in `AI_USE_LOG.md`.
- Work on a branch and open a pull request for the student to merge; never push to `main`.
