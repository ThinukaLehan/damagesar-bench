# IEEE JSTARS manuscript

`main.tex` (with `refs.bib` and `figures/`) is the paper in the standard IEEE journal LaTeX template (`IEEEtran`,
`journal` option), for IEEE JSTARS (target venue agreed with the supervisor on 22 Nov 2026: an IEEE Q1 journal).
IEEE Access was the earlier target; its template is no longer used.

## Build

1. Download the IEEE journal LaTeX template (IEEE Template Selector → Transactions/Journals → LaTeX) and copy
   `IEEEtran.cls` and `IEEEtran.bst` into this folder. They are IEEE's files and are not kept in this repository
   (see `.gitignore`). TeX Live usually ships them already.
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

- Fill every **[ ]** placeholder from the locked run (k01): Section VI-I, Table XIV, the abstract and the conclusion.
- Replace the three `[DOI to be added]` entries in `refs.bib` (plan, code, data) with the Zenodo DOIs.
- References: peer-reviewed versions only (no preprints, 22 Nov 2026); 49 entries, 13 pages, in line with recent JSTARS papers (e.g. Russo et al., 2026: 13 pages).

- Author biographies and photos, ORCID iDs, affiliations and author order (to confirm with the supervisor).
- Rerun `notebooks/reproduce/r01_reproduce_paper.ipynb` on the final commit: the verdict must read `REPRODUCED`.
