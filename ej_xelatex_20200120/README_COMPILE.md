# ABC – Automatic Beep Censor (EJ LaTeX)

This folder contains a LaTeX manuscript formatted for the Engineering Journal (EJ) using XeLaTeX.

## Files
- `abc_senior_project_ej.tex` – Main manuscript using the EJ template styling
- `ej_template.tex` – Original EJ example (reference)
- `ref.bib` – Bibliography entries (placeholders where exact citations are unknown)

## Requirements
- XeLaTeX engine (TeX Live / MiKTeX)
- BibTeX
- Fonts: GaramondNo8 (as referenced by the EJ template). If unavailable, either install it or change the `\setmainfont` line in the .tex file to an available Garamond variant.

## Quick compile (Windows PowerShell)
Run these commands in this folder:

```pwsh
# 1) Compile (XeLaTeX) -> 2) BibTeX -> 3) Compile twice
xelatex abc_senior_project_ej.tex
bibtex abc_senior_project_ej
xelatex abc_senior_project_ej.tex
xelatex abc_senior_project_ej.tex
```

This will produce `abc_senior_project_ej.pdf`.

## Notes
- If images referenced in the EJ template (e.g., `ej_banner.png`, `gm.eps`, `h-bar.png`, author photos) are missing, comment out those includegraphics lines or add your own assets.
- If GaramondNo8 isn't installed, change to an available font, e.g. `\setmainfont{Garamond}` or another serif font.
- The references are placeholders; update `ref.bib` with exact bibliographic details as needed.
