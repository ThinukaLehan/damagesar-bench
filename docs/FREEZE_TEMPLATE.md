# Freeze record (template)

Copy to `docs/FREEZE.md` on the freeze day, fill the four fields, and merge it in the same PR that sets
`freeze_date` in `configs/events.yaml` to the same date. Then tag that merge commit with the tag name below.
The locked-test notebook (k01) refuses to run unless all of this holds (`src/damagesar_bench/freeze.py`).

date: YYYY-MM-DD
tag: prereg-freeze
doi: 10.5281/zenodo.0000000
level: 0.975

- `doi`: the time-stamped deposit of `docs/preregistration.md` (Zenodo or OSF).
- `level`: 0.975 if each of the two primary claims is tested at 2.5% (Bonferroni), else 0.95.
