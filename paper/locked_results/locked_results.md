# Locked confirmatory test: numbers for the paper

Run date 2026-11-22, commit 92fba68; level 0.975; events haiti, turkey, morocco, noto, bata.

## RQ1 (confirmatory): satellite DINOv3 ViT-L minus ImageNet ViT-B, outline, native

- Per-event gains: bata +0.0852, haiti -0.1264, morocco +0.0731, noto -0.0715, turkey +0.0353
- Estimate -0.0009, 97.5% interval [-0.1470, +0.1453], SD 0.0935
- Verdict (decision table, re-derived): **Inconclusive** (row 4)
- Two-sided sign-flip p 1.0000
- Supporting: satellite ahead (bootstrap interval > 0) in 0 of 5; ImageNet in 1 of 5; one-sided sign-flip p 0.5312

## Per-event paired differences (supporting)

| event | n | diff | boot_lower | boot_upper | boot_reliable | delong_se | delong_p | counts_for |
|---|---|---|---|---|---|---|---|---|
| bata | 7390 | 0.0852 | -0.0431 | 0.2153 | True | 0.0177 | 0.0 | neither |
| haiti | 15557 | -0.1264 | -0.2777 | 0.0723 | True | 0.0386 | 0.001 | neither |
| morocco | 5760 | 0.0731 | -0.0751 | 0.2264 | True | 0.0471 | 0.121 | neither |
| noto | 6969 | -0.0715 | -0.1414 | -0.0339 | True | 0.0098 | 0.0 | b |
| turkey | 119280 | 0.0353 | -0.0089 | 0.0781 | True | 0.005 | 0.0 | neither |

## Exploratory comparisons

| label | estimate | lower | upper | verdict | sign_flip_p |
|---|---|---|---|---|---|
| exploratory DINOv3 ViT-B vs ImageNet (outline) | -0.0482 | -0.2751 | 0.1786 | Inconclusive | 0.5 |
| RQ4 DINOv3 vs image-free | 0.0483 | -0.1712 | 0.2677 | Inconclusive | 0.9375 |
| RQ4 building vs surroundings (DINOv3) | 0.0273 | -0.0087 | 0.0633 | Inconclusive | 0.125 |

## RQ3 (exploratory): label-free prevalence, RQ1 model

| encoder | pooling | event | n | true_prevalence | baseline | em | bbse | youden | abstain | reason | estimate | count_estimate | rel_error | baseline_rel_error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov3_vitl16_sat@native | outline | bata | 7390 | 0.049 | 0.163 | 0.0 | 0.0 | 0.023 | True | source tpr - fpr 0.02 < 0.2 |  |  |  | 2.323 |
| dinov3_vitl16_sat@native | outline | haiti | 15557 | 0.008 | 0.163 | 1.0 | 1.0 | 0.023 | True | source tpr - fpr 0.02 < 0.2 |  |  |  | 19.099 |
| dinov3_vitl16_sat@native | outline | morocco | 5760 | 0.01 | 0.163 | 1.0 | 1.0 | 0.023 | True | source tpr - fpr 0.02 < 0.2 |  |  |  | 16.049 |
| dinov3_vitl16_sat@native | outline | noto | 6969 | 0.154 | 0.163 | 0.865 | 0.0 | 0.023 | True | source tpr - fpr 0.02 < 0.2; EM and BBSE disagree by more than 50% |  |  |  | 0.059 |
| dinov3_vitl16_sat@native | outline | turkey | 119280 | 0.041 | 0.163 | 1.0 | 1.0 | 0.023 | True | source tpr - fpr 0.02 < 0.2 |  |  |  | 2.944 |

Image-free control: not blind on these events (31 Oct audit computed a building-count AUROC).
