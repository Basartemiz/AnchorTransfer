#!/usr/bin/env bash
# 04_build_paper.sh — Compile the LaTeX manuscript
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/reproduce/common.sh"
init_repro_context "$REPO_ROOT"

require_command pdflatex
require_command bibtex

cd "$REPO_ROOT/paper"

echo "=== Compiling main paper ==="
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "=== Compiling supplementary ==="
pdflatex -interaction=nonstopmode supplementary.tex
pdflatex -interaction=nonstopmode supplementary.tex

echo "=== Build complete ==="
echo "Output: paper/main.pdf, paper/supplementary.pdf"
