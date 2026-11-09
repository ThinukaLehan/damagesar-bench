# AI-use log

Every use of a generative-AI tool, what it produced, and how it was checked. AI never replaces reading original sources, design decisions, interpretation of results, or authorship of the report.

| Date | Tool | Purpose | Output | How verified |
|---|---|---|---|---|
| 2026-10-31 | Claude | Dataset audit script and run | Audit tables | Re-run in own pipeline on 3 Nov (class shares reproduced) |
| 2026-11-09 | Claude | Repo skeleton: config, data validation, no-data rule, guard, splits, CLI, tests | v0.1.0 | Read line by line; tests on synthetic tiles; `bench validate` re-run on real BRIGHT in Colab and compared with `bright_event_summary.csv` (3 Nov) — **to do** |
