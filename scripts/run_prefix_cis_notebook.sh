#!/usr/bin/env bash
set -euo pipefail

mkdir -p executed_notebooks Figs
jupyter nbconvert   --to notebook   --execute notebooks/02_prefix_cis_figure_s4.ipynb   --output 02_prefix_cis_figure_s4.executed.ipynb   --output-dir executed_notebooks   --ExecutePreprocessor.timeout=-1   --ExecutePreprocessor.cwd="$(pwd)"
