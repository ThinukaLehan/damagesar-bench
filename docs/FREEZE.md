# Freeze record

The analysis plan (docs/preregistration.md) is frozen. The locked-test notebook (k01) refuses to run unless the
fields below, the git tag and freeze_date in configs/events.yaml agree.

date: 2026-11-22
tag: prereg-freeze
doi: 10.5281/zenodo.24443763
level: 0.975
rq1_encoder: dinov3_vitl16_sat

## Deposit

- Zenodo record: https://zenodo.org/records/24443763 (version 1.0, CC BY 4.0, published 22 Nov 2026)
- File: DamageSAR-Bench_preregistration_2026-11-22.pdf, 87,009 bytes, md5 9a6e7a2beeaa84ac908874e69b9e5a79

## Supervisor agreement

- The ten decisions in the plan's decision record were agreed with Dr. Achala Aponsu by phone on 22 Nov 2026.
- Dr. Aponsu approved depositing the plan and proceeding to the locked test by phone on 22 Nov 2026, at about
  22:15 (Asia/Colombo).

## Order of events

1. Plan deposited on Zenodo (DOI above), 22 Nov 2026.
2. This record merged and the merge commit tagged prereg-freeze.
3. Locked test (k01) run once from a commit that contains the tag.
