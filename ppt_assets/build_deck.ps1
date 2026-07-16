$ErrorActionPreference = "Stop"

$deck = "hamiltonian_geometric_results_deck.tex"

if (Get-Command latexmk -ErrorAction SilentlyContinue) {
    latexmk -pdf -interaction=nonstopmode $deck
    exit $LASTEXITCODE
}

if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
    pdflatex -interaction=nonstopmode $deck
    pdflatex -interaction=nonstopmode $deck
    exit $LASTEXITCODE
}

throw "No LaTeX compiler found. Install MiKTeX or TeX Live, then rerun this script from ppt_assets."
