# IEEE Access manuscript

`main.tex` (with `refs.bib` and `figures/`) is the paper in the IEEE Access LaTeX template.

## Build

1. Download the IEEE Access LaTeX template from the IEEE Access author guidelines
   (https://ieeeaccess.ieee.org/guide-for-authors/preparing-your-article/). Copy its class, style, font and logo
   files (`ieeeaccess.cls`, `IEEEtran.cls`, `IEEEtran.bst`, `spotcolor.sty`, the `t1*` font files and maps,
   `logo.png`, `notaglinelogo.png`, `bullet.png`) into this folder. They are IEEE's files and are not kept in this
   repository (see `.gitignore`).
2. Build:

   ```
   pdflatex main && bibtex main && pdflatex main && pdflatex main
   ```

   Overleaf works too: upload this folder and the template files.

## Figures

- `figures/fig_pipeline.tex` → `fig_pipeline.pdf` (Fig. 1): `pdflatex fig_pipeline.tex`
- `figures/make_fig_auroc.py` → `fig_auroc.pdf` (Fig. 2): `python make_fig_auroc.py`. The values in the script
  are the recorded results, identical to `paper/expected_values.yaml`.

## Before submission

- Author biographies and photos, ORCID iDs, affiliations and author order (to confirm with the supervisor).
- Rerun `notebooks/reproduce/r01_reproduce_paper.ipynb` on the final commit: the verdict must read `REPRODUCED`.
