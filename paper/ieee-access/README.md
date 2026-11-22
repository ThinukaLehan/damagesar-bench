# IEEE Access version (generated)

`main.tex` here is **generated** from `../jstars/main.tex` by `make_access.py`: the text, tables, figures and
references are the JSTARS manuscript's; only the front matter (IEEE Access template, author block, ABSTRACT /
INDEX TERMS) and the biography block differ. Edit the JSTARS file, then run:

```
python paper/ieee-access/make_access.py
```

## Build

Copy the official IEEE Access template files into this folder (`ieeeaccess.cls`, `IEEEtran.cls`, `IEEEtran.bst`,
`logo.png`, `notaglinelogo.png`, `bullet.png`; not redistributed here, see `.gitignore`), then:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The references and figures are read from `../jstars/`. Author photos are framed placeholders until the photos exist.
The target venue agreed with the supervisor is IEEE JSTARS; this version exists so the two templates can be compared.
