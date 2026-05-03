#!/usr/bin/env bash
set -euo pipefail

mkdir -p executed_notebooks Figs
jupyter nbconvert   --to notebook   --execute notebooks/01_main_benchmark_analysis.ipynb   --output 01_main_benchmark_analysis.executed.ipynb   --output-dir executed_notebooks   --ExecutePreprocessor.timeout=-1   --ExecutePreprocessor.cwd="$(pwd)"
