# AI-use log

Every use of a generative-AI tool, what it produced, and how it was checked. AI never replaces reading original sources, design decisions, interpretation of results, or authorship of the report.

**Permission:** on 9 Nov 2026 the supervisor (Dr. Achala Aponsu) confirmed that using generative AI in this project is allowed, and encouraged, given the time available. Every use is logged below and disclosed in the README. On 9 Nov 2026, on the supervisor's advice, the author is listed as the sole author and AI is acknowledged as a tool (as IEEE asks for AI-generated content); the `Co-Authored-By: Claude` lines were removed from the first four commit messages and are no longer added. The code in those commits is unchanged. The formal declaration follows the module's required format.

| Date | Tool | Purpose | Output | How verified |
|---|---|---|---|---|
| 2026-10-31 | Claude | Dataset audit script and run | Audit tables | Re-run in own pipeline on 3 Nov (class shares reproduced) |
| 2026-11-09 | Claude | Repo skeleton: config, data validation, no-data rule, guard, splits, CLI, tests | v0.1.0 | Read line by line; tests on synthetic tiles; `bench validate` re-run on real BRIGHT in Colab and compared with `bright_event_summary.csv` (3 Nov) — **to do** |
| 2026-11-09 | Claude | Colab notebooks 01 (instance-label coverage) and 02 (token size, resolution) written and run in the student's Colab | Results in EXPERIMENT_LOG and ADR 0001 | Outputs read by the student; resolution bug (degrees vs metres) found and fixed; round 2 with instance polygons planned |
| 2026-11-09 | Claude | Building-table loader, CLI command, tests, P01 notebook | `data/buildings.py`, `tests/test_buildings.py` | Self-review found and fixed 3 issues before commit (Web Mercator scale, non-square EPSG:4326 pixels, empty-footprint reason); pixel sizes cross-checked with pyproj; synthetic end-to-end check; real-data check pending (P01) |
