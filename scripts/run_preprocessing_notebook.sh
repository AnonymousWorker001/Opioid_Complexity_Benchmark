#!/usr/bin/env bash
set -euo pipefail

mkdir -p executed_notebooks Figs
jupyter nbconvert   --to notebook   --execute notebooks/00_preprocess_ctn0094.ipynb   --output 00_preprocess_ctn0094.executed.ipynb   --output-dir executed_notebooks   --ExecutePreprocessor.timeout=-1   --ExecutePreprocessor.cwd="$(pwd)"
